import argparse
import json
import logging
import sys
from pathlib import Path

import pandas as pd

import cv2

SRC_PATH = Path(__file__).resolve().parents[1]
sys.path.append(str(SRC_PATH))
from src.utils import func_profile, log_handler
from src.visualize import convert_to_dataframe, show_and_save_video


def argparser():
    parser = argparse.ArgumentParser()
    parser.add_argument('-v', '--video', dest='video', required=True)
    parser.add_argument('-o', '--options',
                        dest='options',
                        choices=['detection', 'classification', 'track', 'action'],
                        required=True)
    parser.add_argument('-i', '--input-record', dest='record', required=True)
    parser.add_argument('-c', '--config', dest='config', required=True)
    parser.add_argument('-m', '--mouse', dest='mouse_contours')
    parser.add_argument('--from', dest='from_idx', default=0, type=int)
    parser.add_argument('--show-video', dest='show_video', action='store_true', \
                        help='show video with cv2')
    parser.add_argument('--no-show-video', dest='show_video', action='store_false')
    parser.add_argument('--save-video', dest='save_video', action='store_true', \
                        help='save video with given name')
    parser.add_argument('--no-save-video', dest='save_video', action='store_false')
    parser.add_argument('--pause', dest='pause', action='store_true')
    parser.add_argument('--no-pause', dest='pause', action='store_false')
    parser.add_argument('--output-video', dest='outvideo', default='track.avi')
    parser.set_defaults(show_video=True, save_video=False, pause=False)
    return parser

@func_profile
def main(args):
    logger = logging.getLogger(__name__)
    log_handler(logger, logging.getLogger('src.visualize'))
    logger.info(args)
    record_path = Path(args.record)
    save_video_path = record_path.parent / f'{record_path.stem}_{args.options}.avi'
    save_video_path = save_video_path if args.save_video else None

    with open(args.config, 'r') as f:
        config = json.load(f)['outputs']
    mouse_contours = None
    if args.mouse_contours:
        with open(args.mouse_contours, 'r') as f:
            mouse_contours = json.load(f)
    if args.save_video:
        if not save_video_path.parent.exists():
            save_video_path.parent.mkdir(parents=True)
        save_video_path = str(save_video_path)
    
    if args.options in ['detection', 'classification']:
        with open(args.record, 'r') as f:
            records = f.readlines()
            records = [eval(line) for line in records]
        df = convert_to_dataframe(records, args.options)
        show_and_save_video(args.video, df, config,
                            from_=args.from_idx,
                            show_video=args.show_video,
                            save_video=save_video_path,
                            pause_flag=args.pause)
    elif args.options == 'track':
        df = pd.read_csv(args.record)
        show_and_save_video(args.video, df, config,
                            from_=args.from_idx,
                            mouse_contours=mouse_contours,
                            show_video=args.show_video,
                            save_video=save_video_path,
                            pause_flag=args.pause)
    elif args.options == 'action':
        df = pd.read_csv(args.record)
        if Path(args.record).name != 'fixed_action_paths.csv':
            df = convert_to_dataframe(df, args.options)
            df.to_csv(Path(args.record).with_name('fixed_action_paths.csv'))
        show_and_save_video(args.video, df, config,
                            from_=args.from_idx,
                            show_video=args.show_video,
                            save_video=save_video_path,
                            pause_flag=args.pause)

if __name__ == '__main__':
    main(argparser().parse_args())
