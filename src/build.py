import copy
import json
import logging
from multiprocessing import Pool, cpu_count

import numpy as np

import cv2
from src.match import hungarian_matching
from src.structure import *
from src.utils import func_profile
from tqdm import tqdm

LOGGER = logging.getLogger(__name__)

def _binary_search_less_equal_number(seq, n):
    """[summary] find the maxinmum number which is less equal than n
    - assume the sequence is sorted
    - find the minimum greater number than -1 index to get the result
    
    Arguments:
        seq {[list]} -- list of integer
        n {[int]} -- target number to compare
    """
    if not len(seq) or n is None:
        return -1
    if n < seq[0]:
        return 0
    if n > seq[-1]:
        return len(seq)-1
    l_idx, r_idx = 0, (len(seq)-1)
    while l_idx < r_idx:
        m_idx = l_idx + (r_idx - l_idx) // 2 
        middle_n = seq[m_idx]
        if middle_n < n:
            l_idx = m_idx + 1
        elif middle_n == n:
            return m_idx
        else:
            r_idx = m_idx
    return r_idx if seq[r_idx] == n else max(0, r_idx-1)

def _transform_data_format(records, confidence_threshold=0.0):
    for rid, record in enumerate(records):
        if record[1]:
            record[1] = list(filter(lambda x: x[4] > confidence_threshold, record[1]))
            
            for idx, bbox in enumerate(record[1]):
                bbox[:4] = list(map(int, bbox[:4]))
                bbox = BBox(frame_idx=record[0],
                            pt1=tuple(bbox[0:2][::-1]),
                            pt2=tuple(bbox[2:4][::-1]),
                            confidence=bbox[4],
                            multiclass_result=bbox[5])
                record[1][idx] = bbox
        records[rid] = Frame(frame_idx=record[0], bboxes=record[1])
    return records

@func_profile
def _interpolate_mouse_contour_in_timeline(video: str,
                                           kernel_size: list,
                                           mouse_contours: list,
                                           shift_boundary: int):
    """[summary] calculate mouse contour if the center is shifting in sequence
    
    Arguments:
        video {str} -- video filename
        kernel_size {list} -- kernel size for mouse contour dilation
        mouse_contours {list} -- list of mouse instances
        shift_boundary {int} -- threshld to determine if whether to calc mouse contour 
    """
    _before_number_of_mouse = len(mouse_contours)
    mouse_idx = 0
    while mouse_idx < len(mouse_contours) - 1:
        l_mouse = mouse_contours[mouse_idx]
        r_mouse = mouse_contours[mouse_idx + 1]
        dist = np.linalg.norm(r_mouse.center - l_mouse.center)

        if dist <= shift_boundary:
            mouse_idx += 1
            continue

        new_mouse_idx = l_mouse.frame_idx + (r_mouse.frame_idx - l_mouse.frame_idx) // 2
        if new_mouse_idx == l_mouse.frame_idx:
            mouse_idx += 1
            continue
        
        mouse = calculate_mouse_contour(video, new_mouse_idx, kernel_size)
        mouse_contours.insert(mouse_idx+1, mouse)
    
    _after_number_of_mouse = len(mouse_contours)
    LOGGER.info('Number of mouse from {} to {}'.format(_before_number_of_mouse, _after_number_of_mouse))
    LOGGER.debug('{}'.format([i.frame_idx for i in mouse_contours]))
    return mouse_contours

def calculate_mouse_contour(video: str, frame_idx: int, kernel_size: list):
    cap = cv2.VideoCapture(video)
    cap.set(1, frame_idx)
    success, frame = cap.read()
    mouse = Mouse()
    mouse.update_by_frame_idx(frame_idx, frame, kernel_size)
    cap.release()
    return mouse

def check_on_mouse(trackflow, mouse_contours):
    try:
        _mouse_frame_indexes = [mouse.frame_idx for mouse in mouse_contours]
        for label, flow in trackflow.paths.items():
            for bbox in flow:
                mouse_idx = _binary_search_less_equal_number(_mouse_frame_indexes, bbox.frame_idx)
                mouse = mouse_contours[mouse_idx]
                bbox.behavior['on_mouse'] = mouse.is_touch(bbox)
        trackflow.mouse_cnts = mouse_contours
        trackflow.check_on_mouse = True
    except Exception as e:
        LOGGER.exception(e)

@func_profile
def build_flow(video:str, filename: str, config: str):
    with open(filename, 'r') as f:
        records = f.readlines()
        records = list(map(eval, records))
    with open(config, 'r') as f:
        config = json.load(f)

    with Pool() as pool:
        # multiprocess to pre-calculate mouse contour
        cap = cv2.VideoCapture(video)
        total_frame = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        args = zip([video]*total_frame,
                    range(0, total_frame, config['mouse_cnts_per_nframe']),
                    [config['mouse_cnts_dilate_kernel_size']]*total_frame)
        cap.release()
        LOGGER.info('System total cpu count: {}'.format(cpu_count()))
        LOGGER.info('Video {}, total frame={}'.format(video.split('/')[-1], total_frame))
        mouse_contours = pool.starmap_async(calculate_mouse_contour, args)

        # convert data format
        records = _transform_data_format(records, config['detect_confidence_threshold'])

        # build flow
        reference_bbox = None
        init_assign_id = False
        max_bbox_id = 0
        tracker_blocks = []

        LOGGER.info('Apply matching algorithm to each frame')
        for fid, frame in enumerate(tqdm(records[:-1])):
            # initial bbox assign id
            if not init_assign_id and frame.bboxes:
                for bbox in frame.bboxes:
                    bbox.assign_id = max_bbox_id
                    max_bbox_id += 1
                init_assign_id = True
                reference_bbox = frame.bboxes[0]
            if not init_assign_id:
                continue

            # match tracked bbox and add new untrack bbox
            if records[fid].bbox_count > 0 and records[fid+1].bbox_count > 0:
                match_indexes, match_matrix = hungarian_matching(records[fid], records[fid+1])
                for n_bbox1, n_bbox2 in match_indexes:
                    if match_matrix[n_bbox1][n_bbox2] > config['match_score_threshold']:
                        bbox1 = records[fid].bboxes[n_bbox1]
                        bbox2 = records[fid+1].bboxes[n_bbox2]
                        bbox2.assign_id = bbox1.assign_id

                        # build a list of TrackBlock
                        if not tracker_blocks:
                            block = TrackBlock(bbox1)
                            block.append(bbox2)
                            tracker_blocks.append(block)
                            continue
                        for bid, b in enumerate(tracker_blocks):
                            # find block exist bbox1 and append bbox2
                            if b.tail == bbox1:
                                b.append(bbox2)
                                break
                            # new trackblock and append bbox1, bbox2
                            if bid == len(tracker_blocks)-1 and b.tail != bbox1:
                                block = TrackBlock(bbox1)
                                block.append(bbox2)
                                tracker_blocks.append(block)
                                break

                for next_frame_bbox in [i for i in records[fid+1].bboxes if not i.assign_id]:
                    next_frame_bbox.assign_id = max_bbox_id
                    max_bbox_id += 1
        
        # detect behavior
        mouse_contours = mouse_contours.get()
        mouse_contours = sorted(mouse_contours, key=lambda x: x.frame_idx)
        mouse_contours = _interpolate_mouse_contour_in_timeline(video,
                                                                config['mouse_cnts_dilate_kernel_size'],
                                                                mouse_contours,
                                                                config['mouse_center_shift_boundary'])
        # final result
        trackflow = TrackFlow(reference_bbox.multiclass_result.keys())
        LOGGER.info('Build TrackFlow with {} block'.format(len(tracker_blocks)))
        for block in tqdm(tracker_blocks):
            block.vote_for_label()
            if len(block.bboxes) > config['block_length_threshold'] and \
            block.confidence > config['block_scroe_threshold']:
                trackflow.append_block(block.label, block)
                for bbox in block.bboxes:
                    bbox.block_confidence = block.confidence
        for k, v in trackflow.paths.items():
            v = sorted(v, key=lambda x: x.frame_idx)
        check_on_mouse(trackflow, mouse_contours)

    return trackflow
