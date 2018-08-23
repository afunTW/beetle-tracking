import argparse
import datetime
import logging
import os
import sys
from pathlib import Path
from pprint import pprint

import numpy as np
import pandas as pd

import cv2
from src.utils import func_profile, log_handler
from tqdm import tqdm


def argparser():
    parser = argparse.ArgumentParser()
    parser.add_argument('-v', '--video', dest='video', required=True)
    parser.add_argument('-i', '--observer-file', dest='observer', required=True)
    parser.add_argument('-p', '--paths-file', dest='paths', required=True)
    return parser

def ms_to_hmsf(ms):
    delta = datetime.timedelta(microseconds=ms*1000)
    hmsf = (datetime.datetime.min + delta).strftime('%H:%M:%S.%f')
    return hmsf

def hmsf_to_idx(hmsf, fps):
    # hmsf <datetime.time>
    # get millisecond -> divide 1/15 -> frame idx
    base = datetime.datetime.strptime('00:00:00.000000', '%H:%M:%S.%f')
    hmsf = str(hmsf) if hmsf.microsecond else '{}.000000'.format(str(hmsf))
    hmsf = datetime.datetime.strptime(str(hmsf), '%H:%M:%S.%f')
    delta = hmsf - base
    total_milliseconds = delta.microseconds/1e+6 + delta.seconds
    return int(total_milliseconds * fps)

@func_profile
def main(args):
    logger = logging.getLogger(__name__)
    log_handler(logger)
    logger.info(args)

    outdir = Path(__file__).parent / '../output/path'
    if not outdir.exists():
        logger.info('{} not found, created.'.format(str(outdir)))
        outdir.mkdir(parents=True)
    
    observer_filepath = Path(args.observer)
    paths_filepath = Path(args.paths)
    fixed_video_name = str(paths_filepath.parent).split(os.sep)[-1]
    fixed_observer_savepath = paths_filepath.parent / 'fixed_observer.csv'

    # get fps from video
    cap = cv2.VideoCapture(args.video)
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()
    logger.info('video fps = {}'.format(fps))
    
    # clean observer file
    df_observer = pd.read_csv(observer_filepath)
    df_observer = df_observer[['Time_Relative_hmsf',
                               'Duration_sf',
                               'Subject',
                               'Behavior',
                               'Modifier_1',
                               'Event_Type']]
    df_observer = df_observer.dropna(subset=['Modifier_1'])
    df_observer.columns = ['timestamp_hmsf', 'duration', 'subject', 'behavior', 'target', 'event_type']
    df_observer['label'] = df_observer.apply(lambda row: row['subject'].split(' ')[-1], axis=1)
    df_observer['timestamp_hmsf'] = pd.to_datetime(df_observer['timestamp_hmsf'])
    df_observer['timestamp_hmsf'] = [t.time() for t in df_observer['timestamp_hmsf']]
    df_observer.to_csv(fixed_observer_savepath, index=False)
    logger.info('Complete to clean and save observer file at {}'.format(fixed_observer_savepath))

    # padding the records in duration
    df_padding_event = df_observer.copy().reset_index(drop=True)
    df_padding_event['frame_idx'] = df_padding_event.apply(lambda x: hmsf_to_idx(x['timestamp_hmsf'], fps), axis=1)
    action_segment = []
    for index, row in df_observer.iterrows():
        if row['event_type'] == 'State start':
            logger.info('From row {:02} - {}'.format(index, row.tolist()))
            for end_index, end_row in df_observer.loc[index+1:, ].iterrows():
                check_same_event = end_row['subject': 'target'].tolist() == row['subject': 'target'].tolist()
                check_same_event = check_same_event and end_row['event_type'] == 'State stop'
                if check_same_event:
    
                    # create the DaaFrame to pad the loss record per event
                    # date_range build wiht milliseconds
                    time1 = datetime.datetime.combine(datetime.date.min, row['timestamp_hmsf'])
                    time2 = datetime.datetime.combine(datetime.date.min, end_row['timestamp_hmsf'])
                    delta = time2 - time1
                    ts = pd.date_range(str(row['timestamp_hmsf']),
                                       periods=delta.total_seconds()*1000,
                                       freq='L', closed='right')
                    ts = [t.time() for t in ts.tolist()]
                    ts_data = {
                        'timestamp_hmsf': ts,
                        'duration': [0.0]*len(ts),
                        'subject': [row['subject']]*len(ts),
                        'behavior': [row['behavior']]*len(ts),
                        'target': [row['target']]*len(ts),
                        'event_type': ['State processing']*len(ts),
                        'label': [row['label']]*len(ts)
                    }
                    df_ts = pd.DataFrame(data=ts_data)
                    df_ts['frame_idx'] = df_ts.apply(lambda x: hmsf_to_idx(x['timestamp_hmsf'], fps), axis=1)
                    df_ts.drop_duplicates(subset=['frame_idx'], keep='first', inplace=True)
                    df_padding_event = pd.concat([df_padding_event, df_ts], sort=False)
                    logger.info('To row {:02} - {}\n{}'.format(end_index, end_row.tolist(), '-'*87))
                    break

    df_padding_event.reset_index(drop=True, inplace=True)
    df_padding_event.drop_duplicates(subset=['frame_idx', 'label'], keep='first', inplace=True)
    logger.info('Complete to create the padding dataframe, shape={}'.format(df_padding_event.shape))

    # join action and path
    # observer precise on millisecond, timestamp_ms should be remove the microsecond part
    df_paths = pd.read_csv(paths_filepath)
    df_final = pd.merge(df_paths, df_padding_event, how='left', on=['frame_idx', 'label'])
    logger.info('paths file shape={}'.format(df_paths.shape))
    logger.info('Compete merge to final dataframe, shape={}'.format(df_final.shape))

    # save final result
    action_path_filepath = paths_filepath.parent / 'action_paths.csv'
    df_final.to_csv(action_path_filepath, index=False)

if __name__ == '__main__':
    main(argparser().parse_args())
