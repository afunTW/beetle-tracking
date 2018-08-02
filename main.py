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
    parser.add_argument('--log', dest='log')
    parser.add_argument('--output-video', dest='out_video')
    return parser

@func_profile
def main(args):
    logdir = Path('logs')
    outdir = Path('output')
    videodir = outdir / 'video'
    if not logdir.exists():
        logdir.mkdir(parents=True)
    if not videodir.exists():
        videodir.mkdir(parents=True)
    video_savepath = str(videodir/args.out_video) if args.out_video else None
    logger = logging.getLogger(__name__)
    log_handler(logger, *LOGGERS, logname=str(logdir / args.log) if args.log else None)
    logger.info(args)

    trackflow = build_flow(args.video, args.classification_result, args.config)
    for label, flow in trackflow.paths.items():
        convert_and_output(outdir, label, flow)
    show_tracker_flow(args.video, trackflow, args.config, save_path=video_savepath)

if __name__ == '__main__':
    parser = argparser()
    main(parser.parse_args())
