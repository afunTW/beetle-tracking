"""[summary]

e.g.
python3 main.py \
--video "[CH01] 2017-03-10 20.45.00_x264.avi" \
--detection-result "[CH01] 2017-03-10 20.45.00_x264.txt" \
--config config/default.json \
--log final.log \
--output-video output.avi

Returns:
    [type] -- [description]
"""

import argparse
import logging
import sys
from pathlib import Path

import cv2
from src.build import build_flow
from src.utils import *

LOGGERS = [
    logging.getLogger('src.utils'),
    logging.getLogger('src.build')
]

def argparser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--video', dest='video', required=True)
    parser.add_argument('--classification-result', dest='classification_result', required=True)
    parser.add_argument('--config', dest='config', default='config/default.json')
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

    trackflow = build_flow(args.video, args.classification_result, args.config)
    for label, flow in trackflow.paths.items():
        convert_and_output(trackpath_dir, label, flow)

    video_savepath = None
    if args.save_video:
        videodir = outdir / 'video'
        if not videodir.exists():
            videodir.mkdir(parents=True)
        video_savepath = str(videodir/args.outvideo) if args.outvideo else None
    if args.show_video or args.save_video:
        show_tracker_flow(args.video, trackflow, args.config,
                          from_=args.from_idx,
                          pause=args.pause,
                          show_video=args.show_video,
                          save_video=video_savepath)

if __name__ == '__main__':
    parser = argparser()
    main(parser.parse_args())
