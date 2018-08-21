import argparse
import json
import logging
from pathlib import Path

import pandas as pd

import cv2
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
    parser.add_argument('-c', '--config', dest='config', default='config/default.json')
    parser.add_argument('-m', '--mouse', dest='mouse_contours')
    parser.add_argument('--from', dest='from_idx', default=0, type=int)
    parser.add_argument('--show-video', dest='show_video', action='store_true', help='show video with cv2')
    parser.add_argument('--no-show-video', dest='show_video', action='store_false')
    parser.add_argument('--save-video', dest='save_video', action='store_true', help='save video with given name')
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
    save_video_path = Path(args.video).with_name(args.outvideo) if args.save_video else None

    with open(args.config, 'r') as f:
        config = json.load(f)['outputs']
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
