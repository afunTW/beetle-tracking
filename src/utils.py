import json
import logging
import sys
from datetime import datetime
from functools import wraps

import numpy as np
import pandas as pd

import cv2
from tqdm import tqdm

LOGGER = logging.getLogger(__name__)


def log_handler(*loggers, logname=None):
    formatter = logging.Formatter(
        '%(asctime)s %(filename)12s:L%(lineno)3s [%(levelname)8s] %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S')

    # stream handler
    sh = logging.StreamHandler(sys.stdout)
    sh.setLevel(logging.INFO)
    sh.setFormatter(formatter)

    # file handler
    if logname:
        fh = logging.FileHandler(logname)
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(formatter)

    for logger in loggers:
        if logname:
            logger.addHandler(fh)
        logger.addHandler(sh)
        logger.setLevel(logging.DEBUG)

def func_profile(func):
    @wraps(func)
    def wrapped(*args, **kwargs):
        start_time = datetime.now()
        result = func(*args, **kwargs)
        cost_time = datetime.now() - start_time
        fullname = '{}.{}'.format(func.__module__, func.__name__)
        LOGGER.info('{}[kwargs={}] completed in {}'.format(
            fullname, kwargs, str(cost_time)
        ))
        return result
    return wrapped

def get_label_color(label=None):
    color_map = {
        'X': (215, 255, 0),     # Gold
        'A': (0, 255, 0),       # Green
        'O': (255, 191, 0),     # DeepSkyBlue
        '=': (71, 99, 255),     # Tomato
        'mouse': (54, 54, 54),  # gray81
    }
    return color_map.get(label) or (255, 255, 255)

def bbox_behavior_encoding(behavior):
    encode_array = []
    checked = ['on_mouse']
    for i in checked:
        encode_array.append(int(behavior[i]))
    return encode_array

def convert_and_output(savedir, label, flow):
    savedir = savedir / 'path'
    if not savedir.exists():
        savedir.mkdir(parents=True)
    savepath = str(savedir / '{}_result.csv'.format(label))
    label_result = [[bbox.frame_idx,
                        *bbox.pt1,
                        *bbox.pt2,
                        *bbox.center,
                        *bbox_behavior_encoding(bbox.behavior)] for bbox in flow]
    df = pd.DataFrame(label_result,
                        columns=['frame_idx',
                                'pt1.x', 'pt1.y',
                                'pt2.x', 'pt2.y',
                                'center.x', 'center.y',
                                'on_mouse'])
    df.to_csv(savepath, index=False)

def show_tracker_flow(video, trackerflow, config, save_path=None):
    # preprocess
    cap = cv2.VideoCapture(video)
    out = None
    pause_flag = False
    with open(config, 'r') as f:
        config = json.load(f)['outputs']
    if save_path:
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        resolution = tuple(map(int, (cap.get(3), cap.get(4))))
        out = cv2.VideoWriter(save_path, fourcc, 15, resolution)
    else:
        cv2.namedWindow('show')

    current_ref_mouse_idx, next_ref_mouse_idx = None, None
    if len(trackerflow.mouse_cnts) > 1:
        current_ref_mouse_idx, next_ref_mouse_idx = 0, 1
    elif len(trackerflow.mouse_cnts) == 1:
        current_ref_mouse_idx = 0

    # draw by each frame
    frame_total_length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    for frame_idx in tqdm(range(frame_total_length)):

        # get frame
        cap.set(1, frame_idx)
        success, frame = cap.read()
        if not success:
            LOGGER.warning('Cannot read the {} frame'.format(frame_idx))
            continue
        
        # get the drawing candidate by path_Length
        draw_candidate = {k: [] for k in trackerflow.paths.keys()}
        for k, bboxes in trackerflow.paths.items():
            for bbox in bboxes:
                if bbox.frame_idx > frame_idx:
                    break
                if max(0, frame_idx-config.get('path_length', 1)) < bbox.frame_idx:
                    draw_candidate[k].append(bbox)

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

            for cnt_idx, cnt in enumerate(mouse.contour_extend):
                if cnt_idx == 0:
                    continue
                else:
                    mouse_color = get_label_color(label='mouse')
                    pt1, pt2 = tuple(mouse.contour_extend[cnt_idx-1][0]), tuple(cnt[0])
                    cv2.line(frame, pt1, pt2, mouse_color, 2, cv2.LINE_AA)

        # draw beetle
        for label, paths in draw_candidate.items():
            label_color = get_label_color(label)
            text_color = (255, 255, 255)
            for bbox_idx, bbox in enumerate(paths):

                # link the path
                if bbox_idx > 1 and paths[bbox_idx-1].next == bbox:
                    last_bbox = paths[bbox_idx-1]
                    cv2.line(frame, bbox.center, last_bbox.center, label_color, 1, cv2.LINE_AA)

                # show current bbox information by conofig
                if bbox_idx == len(paths)-1:
                    cv2.putText(frame, label, bbox.center, cv2.FONT_HERSHEY_COMPLEX, 1, label_color, 2)
                    enable_bbox_in_condiction = \
                        config.get('show_highlight_on_mouse', False) and bbox.behavior['on_mouse']
                    if config.get('show_bbox', False) or enable_bbox_in_condiction:
                        cv2.rectangle(frame, bbox.pt1, bbox.pt2, label_color, 2)
                    
                    # show message
                    show_msg = ''
                    if config.get('show_detect_score', False):
                        show_msg += 'detect {:.2f}'.format(bbox.confidence)
                    if config.get('show_class_score', False):
                        show_msg = show_msg + ', ' if show_msg else show_msg
                        show_msg += 'class {:.2f}'.format(bbox.multiclass_result.get(label, 0))
                    if config.get('show_msg_on_mouse', False) and bbox.behavior.get('on_mouse'):
                        show_msg = show_msg + '\n' if show_msg else show_msg
                        show_msg += 'on mouse'
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

        if save_path:
            out.write(frame)
        else:
            cv2.imshow('show', frame)
            k = cv2.waitKey(0) if pause_flag else cv2.waitKey(1)
            if k in [27, ord('q')]:
                cv2.destroyAllWindows()
                break
            elif k == ord('p'):
                pause_flag = not pause_flag
                continue
    
    cap.release()
    if save_path:
        out.release()
    else:
        cv2.destroyAllWindows()