import argparse
import json
import logging
import sys
from pathlib import Path

import pandas as pd

import cv2
from src.build import build_flow
from src.utils import *
from src.visualize import show_and_save_video
from tqdm import tqdm

LOGGERS = [
    logging.getLogger('src.utils'),
    logging.getLogger('src.build')
]

def argparser():
    parser = argparse.ArgumentParser()
    parser.add_argument('-v', '--video', dest='video', required=True)
    parser.add_argument('-i', '--input', dest='input', required=True, help='classification result')
    parser.add_argument('-c', '--config', dest='config', default='config/default.json')
    parser.add_argument('--from', dest='from_idx', default=0, type=int)
    parser.add_argument('--show-video', dest='show_video', action='store_true', help='show video with cv2')
    parser.add_argument('--no-show-video', dest='show_video', action='store_false')
    parser.add_argument('--save-video', dest='save_video', action='store_true', help='save video with given name')
    parser.add_argument('--no-save-video', dest='save_video', action='store_false')
    parser.add_argument('--output-video', dest='outvideo', default='track.avi')
    parser.add_argument('--pause', dest='pause', action='store_true')
    parser.add_argument('--no-pause', dest='pause', action='store_false')
    parser.add_argument('--log', dest='log', default='final.log')
    parser.set_defaults(show_video=False, save_video=True, pause=False)
    return parser

@func_profile
def main(args):
    logdir = Path('logs')
    outdir = Path('output')
    trackpath_dir = outdir / 'path' / Path(args.video).stem
    if not logdir.exists():
        logdir.mkdir(parents=True)
    if not trackpath_dir.exists():
        trackpath_dir.mkdir(parents=True)
    
    logger = logging.getLogger(__name__)
    log_handler(logger, *LOGGERS, logname=str(logdir / args.log) if args.log else None)
    logger.info(args)

    trackflow = build_flow(args.video, args.input, args.config)

    # get timestamp and save path to file
    cap = cv2.VideoCapture(args.video)
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()

    trackflow_all_label = []
    label_col = sorted(trackflow.ref_keys)
    for label, flow in trackflow.paths.items():
        logger.info('get timestamp and convert {}'.format(label))
        for bbox in tqdm(flow):
            bbox.timestamp = bbox.frame_idx / fps * 1000
            label_score = [bbox.multiclass_result.get(l, None) for l in label_col]
            trackflow_all_label.append([
                bbox.frame_idx, bbox.block_id, bbox.timestamp, label,
                bbox.confidence, *label_score, bbox.block_confidence,
                *bbox.pt1, *bbox.pt2, *bbox.center,
                *bbox_behavior_encoding(bbox.behavior)
            ])
    
    trackflow_all_label = sorted(trackflow_all_label, key=lambda x: x[0])
    label_col = ['{}_score'.format(l) for l in label_col]
    trackflow_cols = [
        'frame_idx', 'block_idx', 'timestamp_ms', 'label',
        'detect_score', *label_col, 'block_score',
        'pt1.x', 'pt1.y', 'pt2.x', 'pt2.y', 'center.x', 'center.y', 'on_mouse'
    ]
    path_savepath = str(trackpath_dir / 'paths.csv')
    df_paths = pd.DataFrame(trackflow_all_label, columns=trackflow_cols)
    df_paths.to_csv(path_savepath, index=False)

    # save mouse contours
    mouse_cnts_filepath = str(Path(args.video).parent / '{}_mouse.json'.format(Path(args.video).stem))
    mouse_cnts = {
        str(m.frame_idx): {
            'contour': m.contour.tolist(),
            'contour_extend': m.contour_extend.tolist(),
            'contour_extend_kernel': m.contour_extend_kernel.tolist(),
            'center': m.center.tolist()
        } for m in trackflow.mouse_cnts
    }
    with open(mouse_cnts_filepath, 'w+') as f:
        json.dump(mouse_cnts, f)

    # show and/or save video
    video_savepath = None
    if args.save_video:
        videodir = outdir / 'video'
        if not videodir.exists():
            videodir.mkdir(parents=True)
        video_savepath = str(videodir/args.outvideo) if args.outvideo else None
    if args.show_video or args.save_video:
        with open(args.config, 'r') as f:
            config = json.load(f)
            show_and_save_video(args.video, df_paths, config['outputs'],
                                from_=args.from_idx,
                                show_video=args.show_video,
                                save_video=video_savepath,
                                pause_flag=args.pause)

if __name__ == '__main__':
    parser = argparser()
    main(parser.parse_args())
