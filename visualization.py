import argparse
import logging
import cv2
from src.utils import func_profile, log_handler, draw_on_video

def argparser():
    parser = argparse.ArgumentParser()
    parser.add_argument('-v', '--video', dest='video', required=True)
    parser.add_argument('-o', '--options',
                        dest='options',
                        choices=['detection', 'classification'],
                        required=True)
    parser.add_argument('--from', dest='from_idx', default=0, type=int)
    parser.add_argument('--record-path', dest='record', required=True)
    parser.add_argument('--show-video', dest='show_video', action='store_true', help='show video with cv2')
    parser.add_argument('--no-show-video', dest='show_video', action='store_false')
    parser.add_argument('--save-video', dest='save_video', action='store_true', help='save video with given name')
    parser.add_argument('--no-save-video', dest='save_video', action='store_false')
    parser.add_argument('--pause', dest='pause', action='store_true')
    parser.add_argument('--no-pause', dest='pause', action='store_false')
    parser.add_argument('--output-video', dest='outvideo', default='track.avi')
    parser.set_defaults(show_video=False, save_video=True, pause=False)
    return parser

@func_profile
def main(args):
    logger = logging.getLogger(__name__)
    log_handler(logger)
    logger.info(args)
    if args.options == 'detection':
        with open(args.record, 'r') as f:
            records = f.readlines()
            records = [eval(line) for line in records]
        draw_on_video(args.video, records, 
                      from_=args.from_idx,
                      show_video=args.show_video,
                      save_video=args.save_video,
                      pause_flag=args.pause)
    elif args.options == 'classification':
        with open(args.record, 'r') as f:
            records = f.readlines()
            records = [eval(line) for line in records]
        draw_on_video(args.video, records, 
                      from_=args.from_idx,
                      show_video=args.show_video,
                      save_video=args.save_video,
                      pause_flag=args.pause)

if __name__ == '__main__':
    main(argparser().parse_args())
    