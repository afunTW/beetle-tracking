import json
import logging

import cv2
from src.structure import BBox
from tqdm import tqdm

LOGGER = logging.getLogger(__name__)

def get_label_color(label=None):
    color_map = {
        'X': (215, 255, 0),     # Gold
        'A': (0, 255, 0),       # Green
        'O': (255, 191, 0),     # DeepSkyBlue
        '=': (71, 99, 255),     # Tomato
        'mouse': (54, 54, 54),  # gray81
    }
    return color_map.get(label) or (255, 255, 255)

def show_tracker_flow(video, trackerflow, config,
                      from_=0, pause=False, show_video=False, save_video=None):
    # preprocess
    cap = cv2.VideoCapture(video)
    out = None
    pause_flag = pause
    if save_video:
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        resolution = tuple(map(int, (cap.get(3), cap.get(4))))
        out = cv2.VideoWriter(save_video, fourcc, 15, resolution)
    if show_video:
        cv2.namedWindow('show')

    current_ref_mouse_idx, next_ref_mouse_idx = None, None
    if len(trackerflow.mouse_cnts) > 1:
        current_ref_mouse_idx, next_ref_mouse_idx = 0, 1
    elif len(trackerflow.mouse_cnts) == 1:
        current_ref_mouse_idx = 0

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
        draw_candidate = {k: [] for k in trackerflow.blocks.keys()}
        for k, blocks in trackerflow.blocks.items():
            target_path_length = config.get('path_length', 0)
            target_index_range = set(range(frame_idx-target_path_length, frame_idx+1))
            for block in blocks:
                if block.frame_idx_set & target_index_range:
                    draw_candidate[block.label].append(block)

        # draw mouse
        if current_ref_mouse_idx is not None:
            # determine thr reference mouse
            current_ref_mouse = trackerflow.mouse_cnts[current_ref_mouse_idx]
            next_ref_mouse = trackerflow.mouse_cnts[next_ref_mouse_idx] if next_ref_mouse_idx else None
            mouse = current_ref_mouse
            
            if next_ref_mouse:
                if frame_idx >= next_ref_mouse.frame_idx:
                    current_ref_mouse_idx = next_ref_mouse_idx
                    next_ref_mouse_idx = min(len(trackerflow.mouse_cnts)-1, next_ref_mouse_idx+1)
                    mouse = next_ref_mouse

            for cnt_idx, cnt in enumerate(mouse.contour_extend[1:]):
                    mouse_color = get_label_color(label='mouse')
                    pt1, pt2 = tuple(mouse.contour_extend[cnt_idx-1][0]), tuple(cnt[0])
                    cv2.line(frame, pt1, pt2, mouse_color, 2, cv2.LINE_AA)

        # draw beetle
        for label, blocks in draw_candidate.items():
            label_color = get_label_color(label)
            text_color = (255, 255, 255)
            
            # draw path
            for block_idx, block in enumerate(blocks):
                
                # draw start point
                if config.get('show_start_bbox', False) and block.bboxes:
                    cv2.rectangle(frame, block.head.pt1, block.head.pt2, label_color, 2)
                
                # draw path
                target_path_length = config.get('path_length', 0)
                target_index_range = set(range(frame_idx-target_path_length, frame_idx+1))
                candidate_bboxes = [b for b in block.bboxes if b.frame_idx in target_index_range]
                for bbox_idx, bbox in enumerate(candidate_bboxes):
                    if bbox_idx > 1:
                        cv2.line(frame, bbox.prev.center, bbox.center, label_color, 1, cv2.LINE_AA)
                    
                    # show current bbox information by config
                    if bbox.frame_idx == frame_idx:
                        cv2.putText(frame, label, bbox.center, cv2.FONT_HERSHEY_COMPLEX, 1, label_color, 2)
                        enable_bbox_in_condiction = \
                            config.get('show_highlight_on_mouse', False) and bbox.behavior['on_mouse']
                        if config.get('show_bbox', False) or enable_bbox_in_condiction:
                            cv2.rectangle(frame, bbox.pt1, bbox.pt2, label_color, 2)

                        # show message
                        show_msg = ''
                        if config.get('show_detect_score', False) and bbox.confidence:
                            show_msg += 'detect {:.2f}'.format(bbox.confidence)
                        if config.get('show_class_score', False):
                            show_msg = show_msg + ', ' if show_msg else show_msg
                            show_msg += 'class {:.2f}'.format(bbox.multiclass_result.get(label, 0))
                        if config.get('show_block_score', False) and bbox.block_confidence:
                            show_msg = show_msg + '\n' if show_msg else show_msg
                            show_msg += 'block {:.2f}'.format(bbox.block_confidence)
                        if config.get('show_msg_on_mouse', False) and bbox.behavior.get('on_mouse'):
                            show_msg = show_msg + '\n' if show_msg else show_msg
                            show_msg += 'on mouse'
                        if show_msg:
                            for lid, msg in enumerate(show_msg.split('\n')[::-1]):
                                show_x, show_y = bbox.pt1
                                show_y -= lid*15 + 5
                                show_pts = (show_x, show_y)
                                cv2.putText(frame, msg, show_pts, cv2.FONT_HERSHEY_COMPLEX, 0.5, label_color, 1)

        show_frame_msg = 'frame ({}/{}) {}'.format(
            frame_idx, frame_total_length, 'PAUSE' if pause_flag else '')
        cv2.putText(frame, show_frame_msg, (50, 50), cv2.FONT_HERSHEY_COMPLEX, 1, text_color, 2)

        if save_video:
            out.write(frame)
        if show_video:
            cv2.imshow('show', frame)
            k = cv2.waitKey(0) if pause_flag else cv2.waitKey(1)
            if k in [27, ord('q')]:
                cv2.destroyAllWindows()
                break
            elif k == ord('p'):
                pause_flag = not pause_flag
                continue
    
    cap.release()
    if save_video:
        out.release()
    if show_video:
        cv2.destroyAllWindows()

def draw_on_video(video, records, from_=0, show_video=False, save_video=None, pause_flag=False):
    # preprocess
    cap = cv2.VideoCapture(video)
    out = None
    pause_flag = pause_flag
    if save_video:
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        resolution = tuple(map(int, (cap.get(3), cap.get(4))))
        out = cv2.VideoWriter(save_video, fourcc, 15, resolution)
    if show_video:
        cv2.namedWindow('show')

    # draw by each frame
    frame_total_length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    for frame_idx in tqdm(range(from_, frame_total_length)):

        # get frame
        cap.set(1, frame_idx)
        success, frame = cap.read()
        if not success:
            LOGGER.warning('Cannot read the {} frame'.format(frame_idx))
            continue

        # draw beetle
        paths = records[frame_idx][-1]
        text_color = (255, 255, 255)
        for bbox_idx, bbox in enumerate(paths):
            # convert to BBox structure
            multiclass_result = None
            if isinstance(bbox[-1], dict) and len(bbox) == 6:
                multiclass_result = bbox[-1]
            bbox[:4] = list(map(int, bbox[:4]))
            bbox = BBox(frame_idx=frame_idx,
                        pt1=tuple(bbox[0:2][::-1]),
                        pt2=tuple(bbox[2:4][::-1]),
                        confidence=bbox[4],
                        multiclass_result=multiclass_result)
            
            # scroe1=detection confidence, scroe2=classification confidence 
            score1 = bbox.confidence
            label, score2 = bbox.classification_label if multiclass_result else ('', None)
            label_color = get_label_color(label)

            # show current bbox information by conofig
            cv2.putText(frame, label, bbox.center, cv2.FONT_HERSHEY_COMPLEX, 1, label_color, 2)
            cv2.rectangle(frame, bbox.pt1, bbox.pt2, label_color, 2)
            
            # show message
            show_msg = 'detect {:.2f}'.format(score1)
            if score2:
                show_msg = show_msg + ', ' if show_msg else show_msg
                show_msg += 'class {:.2f}'.format(score2)
            if show_msg:
                for lid, msg in enumerate(show_msg.split('\n')[::-1]):
                    show_x, show_y = bbox.pt1
                    show_y -= lid*15 + 5
                    show_pts = (show_x, show_y)
                    cv2.putText(frame, msg, show_pts, cv2.FONT_HERSHEY_COMPLEX, 0.5, label_color, 1)

        cv2.putText(
            frame, 'frame ({}/{}) {}'.format(frame_idx, frame_total_length, 'PAUSE' if pause_flag else ''),
            (50, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 2
        )

        if save_video:
            out.write(frame)
        if show_video:
            cv2.imshow('show', frame)
            k = cv2.waitKey(0) if pause_flag else cv2.waitKey(1)
            if k in [27, ord('q')]:
                cv2.destroyAllWindows()
                break
            elif k == ord('p'):
                pause_flag = not pause_flag
                continue
    
    cap.release()
    if save_video:
        out.release()
    if show_video:
        cv2.destroyAllWindows()
