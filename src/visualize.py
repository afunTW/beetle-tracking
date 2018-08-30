import logging

import pandas as pd

import cv2
from tqdm import tqdm
from src.structure import BBox

LOGGER = logging.getLogger(__name__)

def _merge_bbox(pt1_lt, pt1_rb, pt2_lt, pt2_rb):
    lt = (min(pt1_lt[0], pt2_lt[0]), min(pt1_lt[1], pt2_lt[1]))
    rb = (max(pt1_rb[0], pt2_rb[0]), max(pt1_rb[1], pt2_rb[1]))
    return lt, rb

def _get_bbox_message(row, config):
    show_msg = ''
    detect_score = float(row['detect_score']) if 'detect_score' in row else None
    label_col = '{}_score'.format(row['label']) if 'label' in row else None
    label_score = float(row[label_col]) if label_col and label_col in row else None
    block_score = float(row['block_score']) if 'block_score' in row else None
    is_on_mouse = int(row['on_mouse']) if 'on_mouse' in row else None
    msg_col = row['msg'] if 'msg' in row else None
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
    if config.get('show_msg_column', False) and msg_col:
        show_msg = show_msg + '\n' if show_msg else show_msg
        show_msg += msg_col
    return show_msg

def _get_center(row):
    col1 = ['center.x', 'center.y']
    col2 = ['center1.x', 'center1.y', 'center2.x', 'center2.y']
    is_col1 = [c in row for c in col1]
    is_col2 = [c in row for c in col2]
    if all(is_col1):
        return tuple(int(row[c]) for c in col1)
    if all(is_col2):
        return tuple(int(row[c]) for c in col2[:2]), tuple(int(row[c]) for c in col2[2:])

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
        for bbox in bboxes:
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

def _convert_action(df):
    results_cols = [
        'frame_idx', 'action_idx', 'pt1.x', 'pt1.y', 'pt2.x', 'pt2.y',
        'center1.x', 'center1.y', 'center2.x', 'center2.y',
        'label', 'behavior', 'target_label', 'msg'
    ]
    data = []
    df_action = 'behavior' in df and df.loc[~df.behavior.isna()].copy()
    df_action['target_label'] = df_action['target'].apply(lambda x: x.split(' ')[-1])
    for name, group in tqdm(df_action.groupby('frame_idx')):
        _checked_in_group = []
        for row_idx, row in group.iterrows():
            t_row = df.loc[(df.frame_idx == row['frame_idx']) & (df.label == row['target_label'])]
            if not t_row.empty:
                # data preprocess and check whether available
                t_row = t_row.reset_index(drop=True).loc[0, :].fillna('')
                if t_row['label'] in _checked_in_group:
                    continue

                # preprare data and append
                subject_pt1 = (row['pt1.x'], row['pt1.y'])
                subject_pt2 = (row['pt2.x'], row['pt2.y'])
                target_pt1 = (t_row['pt1.x'], t_row['pt1.y'])
                target_pt2 = (t_row['pt2.x'], t_row['pt2.y'])
                merge_pt1, merge_pt2 = _merge_bbox( \
                    subject_pt1, subject_pt2, target_pt1, target_pt2)
                subject_center = (row['center.x'], row['center.y'])
                target_center = (t_row['center.x'], t_row['center.y'])

                # message
                event_state = row['event_type'].split(' ')[-1]
                msg = '{} {} {} - {}'.format( \
                    row['label'], row['behavior'], row['target_label'], event_state)
                if 'behavior' in t_row and t_row['behavior']:
                    target_label = t_row['target'].split(' ')[-1]
                    event_state = t_row['event_type'].split(' ')[-1]
                    msg += '\n{} {} {} - {}'.format( \
                        t_row['label'], t_row['behavior'], target_label, event_state)

                data.append([
                    row['frame_idx'], row['action_idx'],
                    *merge_pt1, *merge_pt2, *subject_center, *target_center,
                    row['label'], row['behavior'], row['target_label'], msg
                ])
                _checked_in_group += [row['label'], row['target_label']]

    return pd.DataFrame(data, columns=results_cols)

def convert_to_dataframe(records, option):
    cols, data = [], None
    if option == 'detection':
        cols, data = _convert_detection(records)
    elif option == 'classification':
        cols, data = _convert_classification(records)
    elif option == 'action':
        return _convert_action(records)
    return pd.DataFrame(data, columns=cols)

def draw_mouse_contours(frame, frame_idx, mouse_contours, mouse_idx, current_key):
    """[summary]
    
    Arguments:
        frame {np.ndarray} -- target image
        mouse_contours {dict} -- each mouse contours
        mouse_idx {list} -- list of mouse contours keys
        current_id {int} -- current mouse contours keys index
    """
    next_key = min(mouse_idx.index(current_key)+1, len(mouse_idx)-1)
    next_key = mouse_idx[next_key]
    next_key = next_key if next_key != current_key else None

    current_mouse = mouse_contours[current_key]
    next_mouse = mouse_contours[next_key] if next_key else None
    mouse = current_mouse
    
    if next_mouse and frame_idx >= int(next_key):
        current_key = next_key
        next_key = min(mouse_idx.index(next_key)+1, len(mouse_idx)-1)
        next_key = mouse_idx[next_key]
        mouse = next_mouse

    for cnt_idx, cnt in enumerate(mouse['contour_extend'][1:]):
        mouse_color = _get_label_color(label='mouse')
        pt1, pt2 = tuple(mouse['contour_extend'][cnt_idx-1][0]), tuple(cnt[0])
        cv2.line(frame, pt1, pt2, mouse_color, 2, cv2.LINE_AA)
    
    return current_key, frame

def show_and_save_video(video, records, config,
                        mouse_contours=None,
                        from_=0,
                        show_video=False,
                        save_video=None,
                        pause_flag=False):
    """[summary]

    draw path, label and text to frame based on pd.DataFrame
    
    Arguments:
        video {str} -- video path to read frames
        records {pd.DataFrame} -- different input source, needs different preprocess
        config {dict} -- configuration about drawing
    
    Keyword Arguments:
        show_video {bool} -- whether to show video (local) (default: {False})
        save_video {[type]} -- file path to save result (default: {None})
        pause_flag {bool} -- default pause flag (default: {False})
    """
    LOGGER.info('show_video {}, save_video {}'.format(show_video, save_video))
    LOGGER.info('config - {}'.format(config))

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
    current_ref_mouse_idx, mouse_idx = None, None
    if isinstance(mouse_contours, dict):
        mouse_idx = sorted(map(int, mouse_contours.keys()))
        mouse_idx = list(map(str, mouse_idx))
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
            current_ref_mouse_idx, frame = draw_mouse_contours(
                frame, frame_idx, mouse_contours, mouse_idx, current_ref_mouse_idx
            )
        
        # draw beetle
        groupby_col = 'block_idx' if 'block_idx' in records else None
        iter_df = df_candidate.groupby(groupby_col) if groupby_col else None
        iter_df = [('all', df_candidate)] if iter_df is None else iter_df
        text_color = (255, 255, 255)
        for name, group in iter_df:
            group.reset_index(drop=True, inplace=True)
            for row_idx, row in group.iterrows():
                decide_color_col = row['label'] if 'label' in records else None
                label_color = _get_label_color(decide_color_col)
                pts = pd.to_numeric(row['pt1.x':'pt2.y'], downcast='integer')
                pt1, pt2 = (pts['pt1.x'], pts['pt1.y']), (pts['pt2.x'], pts['pt2.y'])
                center = _get_center(row)

                # draw start point
                if config.get('show_start_bbox', False) and row_idx == 0:
                    cv2.rectangle(frame, pt1, pt2, label_color, 2)

                # draw path
                if row_idx > 0 and groupby_col:
                    prev_row = group.loc[row_idx-1, :]
                    prev_center = _get_center(prev_row)
                    if isinstance(prev_center[0], int) and isinstance(center[0], int):
                        cv2.line(frame, prev_center, center, label_color, 1, cv2.LINE_AA)
                    elif isinstance(prev_center[0], tuple) and isinstance(center[0], tuple):
                        color1 = _get_label_color(row['label'])
                        color2 = _get_label_color(row['target_label'])
                        cv2.line(frame, prev_center[0], center[0], color1, 1, cv2.LINE_AA)
                        cv2.line(frame, prev_center[1], center[1], color2, 1, cv2.LINE_AA)

                if row['frame_idx'] == frame_idx:
                    cond_enable_bbox = config.get('show_highlight_on_mouse', False) and \
                                        'on_mouse' in group and int(row['on_mouse'])
                    if isinstance(center[0], int) and 'label' in row:
                        cv2.putText(frame, row['label'], center, \
                            cv2.FONT_HERSHEY_COMPLEX, 1, label_color, 2)
                    elif isinstance(center[0], tuple) and 'label' in row and 'target_label' in row:
                        color1 = _get_label_color(row['label'])
                        color2 = _get_label_color(row['target_label'])
                        cv2.putText(frame, row['label'], center[0], \
                            cv2.FONT_HERSHEY_COMPLEX, 1, color1, 2)
                        cv2.putText(frame, row['target_label'], center[1], \
                            cv2.FONT_HERSHEY_COMPLEX, 1, color2, 2)
                    if config.get('show_bbox', False) or cond_enable_bbox:
                        cv2.rectangle(frame, pt1, pt2, label_color, 2)
                    
                    # show message
                    show_msg = _get_bbox_message(row, config)
                    if show_msg:
                        for lid, msg in enumerate(show_msg.split('\n')[::-1]):
                            show_x, show_y = pt1
                            show_y -= lid*15 + 5
                            show_pts = (show_x, show_y)
                            cv2.putText(frame, msg, show_pts, \
                                cv2.FONT_HERSHEY_COMPLEX, 0.5, label_color, 1)
    
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
