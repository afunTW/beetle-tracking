import json
import logging
from functools import wraps

import pandas as pd

import cv2
from src.structure import BBox
from tqdm import tqdm

LOGGER = logging.getLogger(__name__)

def _get_label_color(label=None):
    color_map = {
        'X': (215, 255, 0),     # Gold
        'A': (0, 255, 0),       # Green
        'O': (255, 191, 0),     # DeepSkyBlue
        '=': (71, 99, 255),     # Tomato
        'mouse': (54, 54, 54),  # gray81
    }
    return color_map.get(label) or (255, 255, 255)

def _convert_to_bboxes(records):
    LOGGER.info('convert to bboxes')
    data = []
    for record in tqdm(records):
        frame_idx, bboxes = record
        for bbox_idx, bbox in enumerate(bboxes):
            multiclass_result = None
            if isinstance(bbox[-1], dict) and len(bbox) == 6:
                multiclass_result = bbox[-1]
            bbox[:4] = list(map(int, bbox[:4]))
            bbox = BBox(frame_idx=frame_idx,
                        pt1=tuple(bbox[0:2][::-1]),
                        pt2=tuple(bbox[2:4][::-1]),
                        confidence=bbox[4],
                        multiclass_result=multiclass_result)
            data.append(bbox)
    return data

def _convert_detection(records):
    LOGGER.info('convert detectino data')
    bboxes = _convert_to_bboxes(records)
    cols = [
        'frame_idx', 'detect_score',
        'pt1.x', 'pt1.y', 'pt2.x', 'pt2.y',
        'center.x', 'center.y'
    ]
    data = []
    for bbox in tqdm(bboxes):
        data.append([
            bbox.frame_idx, bbox.confidence,
            *bbox.pt1, *bbox.pt2, *bbox.center
        ])
    return cols, data

def _convert_classification(records):
    LOGGER.info('convert classification data')
    bboxes = _convert_to_bboxes(records)
    labels = sorted(bboxes[0].multiclass_result.keys())
    labels_col = ['{}_score'.format(label) for label in labels]
    cols = [
        'frame_idx', 'detect_score', *labels_col,
        'pt1.x', 'pt1.y', 'pt2.x', 'pt2.y',
        'center.x', 'center.y', 'label'
    ]
    data = []
    for bbox in tqdm(bboxes):
        label_score = [bbox.multiclass_result.get(i, None) for i in labels]
        data.append([
            bbox.frame_idx, bbox.confidence, *label_score,
            *bbox.pt1, *bbox.pt2, *bbox.center, bbox.classification_label[0]
        ])
    return cols, data

def convert_to_dataframe(records, option):
    cols, data = [], None
    if option == 'detection':
        cols, data = _convert_detection(records)
    elif option == 'classification':
        cols, data = _convert_classification(records)
    return pd.DataFrame(data, columns=cols)

def show_and_save_video(video, records, config,
                        mouse_contours=None,
                        from_=0,
                        show_video=False,
                        save_video=None,
                        pause_flag=False):
    """[summary]
    
    Arguments:
        video {str} -- video path to read frames
        records {pd.DataFrame} -- different input source, needs different preprocess
        config {dict} -- configuration about drawing
    
    Keyword Arguments:
        show_video {bool} -- whether to show video (local) (default: {False})
        save_video {[type]} -- file path to save result (default: {None})
        pause_flag {bool} -- default pause flag (default: {False})
    """
    # video preprocess
    cap = cv2.VideoCapture(video)
    video_writer = None
    pause_flag = pause_flag
    if show_video:
        cv2.namedWindow('show')
    if isinstance(save_video, str):
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        resolution = tuple(map(int, (cap.get(3), cap.get(4))))
        video_writer = cv2.VideoWriter(save_video, fourcc, 15, resolution)

    # check mouse
    current_ref_mouse_idx, next_ref_mouse_idx, mouse_idx = None, None, None
    if isinstance(mouse_contours, dict):
        mouse_idx = sorted(mouse_contours.keys(), lambda x: int(x))
        mouse_idx = list(map(str, mouse_idx))
        if len(mouse_contours.keys()) > 1:
            current_ref_mouse_idx, next_ref_mouse_idx = mouse_idx[0], mouse_idx[1]
        elif len(mouse_contours.keys()) == 1:
            current_ref_mouse_idx = mouse_idx[0]

    # draw by each frame
    frame_total_length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    for frame_idx in tqdm(range(from_, frame_total_length)):

        # get frame
        cap.set(1, frame_idx)
        success, frame = cap.read()
        if not success:
            LOGGER.warning('Cannot read the {} frame'.format(frame_idx))
            continue
        
        # get the drawing candidate by path_Length
        target_path_length = config.get('path_length', 0)
        target_frame_idx = max(0, frame_idx - target_path_length)
        df_candidate = records[(target_frame_idx <= records.frame_idx) & \
                               (records.frame_idx <= frame_idx)].copy()
        
        # draw mouse
        if current_ref_mouse_idx:
            # determine thr reference mouse
            current_ref_mouse = mouse_contours[current_ref_mouse_idx]
            next_ref_mouse = mouse_contours[next_ref_mouse_idx] if next_ref_mouse_idx else None
            mouse = current_ref_mouse
            
            if next_ref_mouse:
                if frame_idx >= next_ref_mouse.frame_idx:
                    current_ref_mouse_idx = next_ref_mouse_idx
                    next_ref_mouse_idx = min(len(mouse_idx)-1, next_ref_mouse_idx+1)
                    mouse = next_ref_mouse

            for cnt_idx, cnt in enumerate(mouse['contour_extend'][1:]):
                    mouse_color = _get_label_color(label='mouse')
                    pt1, pt2 = tuple(mouse['contour_extend'][cnt_idx-1][0]), tuple(cnt[0])
                    cv2.line(frame, pt1, pt2, mouse_color, 2, cv2.LINE_AA)
        
        # draw beetle
        groupby_col = 'block_idx' if 'block_idx' in records else None
        iter_df = (groupby_col and df_candidate.groupby(groupby_col)) or None
        iter_df = [('all', df_candidate)] if iter_df is None else iter_df
    
        # draw path
        text_color = (255, 255, 255)
        for name, group in iter_df:
            group.reset_index(drop=True, inplace=True)
            for row_idx, row in group.iterrows():
                decide_color_col = row['label'] if 'label' in records else None
                label_color = _get_label_color(decide_color_col)
                pts = pd.to_numeric(row['pt1.x':'center.y'], downcast='integer')
                pt1, pt2 = (pts['pt1.x'], pts['pt1.y']), (pts['pt2.x'], pts['pt2.y'])
                center = (pts['center.x'], pts['center.y'])

                # draw start point
                if config.get('show_start_bbox', False) and row_idx == 0:
                    cv2.rectangle(frame, pt1, pt2, label_color, 2)

                # draw path
                if row_idx > 0 and groupby_col:
                    prev_row = group.loc[row_idx-1, :]
                    prev_center = (prev_row['center.x'], prev_row['center.y'])
                    cv2.line(frame, prev_center, center, label_color, 1, cv2.LINE_AA)

                if row['frame_idx'] == frame_idx:
                    cond_enable_bbox = config.get('show_highlight_on_mouse', False) and \
                                        'on_mouse' in group and \
                                        int(row['on_mouse'])
                    if 'label' in row:
                        cv2.putText(frame, row['label'], center, cv2.FONT_HERSHEY_COMPLEX, 1, label_color, 2)
                    if config.get('show_bbox', False) or cond_enable_bbox:
                        cv2.rectangle(frame, pt1, pt2, label_color, 2)
                    
                    # show message
                    show_msg = ''
                    detect_score = float(row['detect_score']) if 'detect_score' in group else None
                    label_col = '{}_score'.format(row['label']) if 'label' in row else None
                    label_score = float(row[label_col]) if label_col and label_col in group else None
                    block_score = float(row['block_score']) if 'block_score' in group else None
                    is_on_mouse = int(row['on_mouse']) if 'on_mouse' in group else None
                    if config.get('show_detect_score', False) and detect_score:
                        show_msg += 'detect {:.2f}'.format(detect_score)
                    if config.get('show_class_score', False) and label_score:
                        show_msg = show_msg + ', ' if show_msg else show_msg
                        show_msg += 'class {:.2f}'.format(label_score)
                    if config.get('show_block_score', False) and block_score:
                        show_msg = show_msg + '\n' if show_msg else show_msg
                        show_msg += 'block {:.2f}'.format(block_score)
                    if config.get('show_msg_on_mouse', False) and is_on_mouse:
                        show_msg = show_msg + '\n' if show_msg else show_msg
                        show_msg += 'on mouse'
                    if show_msg:
                        for lid, msg in enumerate(show_msg.split('\n')[::-1]):
                            show_x, show_y = pt1
                            show_y -= lid*15 + 5
                            show_pts = (show_x, show_y)
                            cv2.putText(frame, msg, show_pts, cv2.FONT_HERSHEY_COMPLEX, 0.5, label_color, 1)
    
        # show frame information
        pause_msg = 'PAUSE' if pause_flag else ''
        frame_msg = 'frame ({}/{}) {}'.format(frame_idx, frame_total_length, pause_msg)
        cv2.putText(frame, frame_msg, (50, 50), cv2.FONT_HERSHEY_COMPLEX, 1, text_color, 2)

        if show_video:
            cv2.imshow('show', frame)
            k = cv2.waitKey(0) if pause_flag else cv2.waitKey(1)
            if k in [27, ord('q')]:
                cv2.destroyAllWindows()
                break
            elif k == ord('p'):
                pause_flag = not pause_flag
                continue
        save_video and video_writer.write(frame)
    
    cap.release()
    show_video and cv2.destroyAllWindows()
    save_video and video_writer.release()
