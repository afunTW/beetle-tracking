import argparse
import logging
import sys
from pathlib import Path

import cv2
import pandas as pd
from tqdm import tqdm

SRC_PATH = Path(__file__).resolve().parents[1]
sys.path.append(str(SRC_PATH))
from src.utils import func_profile, log_handler
from src.visualize import convert_to_dataframe


def argparser():
    parser = argparse.ArgumentParser()
    parser.add_argument('-v', '--video', dest='video', required=True,\
                        help='source to clip as sequence')
    parser.add_argument('-i', '--input', dest='input', required=True,\
    help='action_pacth.csv')
    return parser

@func_profile
def main(args):
    logger = logging.getLogger(__name__)
    log_handler(logger)
    logger.info(args)

    # video
    video_savepath = Path(args.video)
    cap = cv2.VideoCapture(str(video_savepath))

    # generate with aid and save the clips
    df_action = pd.read_csv(args.input)
    df_action_fixed = convert_to_dataframe(df_action, 'action')
    availabel_action_idx = [i for i in df_action.action_idx.unique() if i >= 0]
    for aid in tqdm(availabel_action_idx):
        # save records
        df_tmp = df_action.loc[df_action.action_idx == aid].copy().reset_index()
        df_tmp_fixed = df_action_fixed.loc[df_action_fixed.action_idx == aid]
        if df_tmp.shape[1] < 5 or df_tmp_fixed.empty:
            continue
        clip_filename = '{}_{}'.format(aid, df_tmp.behavior.unique().tolist()[0])
        clip_savepath = video_savepath.parent / 'clips'/ clip_filename
        if not clip_savepath.exists():
            clip_savepath.mkdir(parents=True)
        df_tmp.to_csv(clip_savepath / 'action_paths.csv', index=False)
        df_tmp_fixed.to_csv(clip_savepath / 'action_paths_fixed.csv', index=False)

        # save clips images
        for row_idx, row in df_tmp_fixed.iterrows():
            cap.set(1, row['frame_idx'])
            success, frame = cap.read()
            if not success:
                logger.exception('frame #{} is not available to read'.format(row['frame_idx']))
                continue
            pt1 = (row['pt1.x'], row['pt1.y'])
            pt2 = (row['pt2.x'], row['pt2.y'])
            img = frame[pt1[1]:pt2[1], pt1[0]:pt2[0]]
            cv2.imwrite(str(clip_savepath / '{}.jpg'.format(row_idx)), img)

if __name__ == '__main__':
    main(argparser().parse_args())
