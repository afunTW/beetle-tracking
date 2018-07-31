import argparse
import logging
from itertools import combinations
from pathlib import Path

import pandas as pd
import numpy as np

from src.utils import func_profile, log_handler
from src.structure import BBox

def argparser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output-path', dest='output_dir', default='output/path')
    parser.add_argument('--sliding-window-length', dest='win_len', default=5, type=int)
    parser.add_argument('--activate-window-length', dest='win_threshold', default=3, type=int)
    return parser

def calc_distance(row):
    pt1 = tuple(map(int, (row['center.x_l'], row['center.y_l'])))
    pt2 = tuple(map(int, (row['center.x_r'], row['center.y_r'])))
    dist = np.linalg.norm(np.array(pt1)-np.array(pt2), ord=2)
    return dist

def check_overlap(row, lrect_col, rrect_col):
    lrect_pts = list(map(int, row[lrect_col]))
    rrect_pts = list(map(int, row[rrect_col]))
    lbbox = BBox(row['frame_idx'], (lrect_pts[0:2]), (lrect_pts[2:4]), None, None)
    rbbox = BBox(row['frame_idx'], (rrect_pts[0:2]), (rrect_pts[2:4]), None, None)
    iou = lbbox.calc_iou(rbbox)
    return iou

def sliding_check_overlap(values, win_threshold):
    return np.count_nonzero(values) > win_threshold

@func_profile
def main(args):
    logger = logging.getLogger(__name__)
    log_handler(logger)
    logger.info(args)

    outputdir = Path(args.output_dir)
    files = list(outputdir.glob('*_result.csv'))
    trackflow = {}

    # read all csv
    from collections import Counter
    for f in files:
        label = str(f.stem).split('_')[0]
        trackflow[label] = pd.read_csv(f)
        logger.info('label {}, shape {}'.format(label, trackflow[label].shape))

    # calc distance ofr each
    for label1, label2 in combinations(trackflow.keys(), 2):
        merge_df = pd.merge(trackflow[label1], trackflow[label2],
                            suffixes=('_l', '_r'),
                            on='frame_idx',
                            how='outer').dropna().reset_index(drop=True)
        merge_df.index = pd.to_datetime(merge_df.frame_idx, unit='s')
        merge_df.index.name = 'frame_time'
        logger.info('merge {} and {}, shape {}'.format(label1, label2, merge_df.shape))
        
        # calc basic distance and overlap area
        col_dist = 'dist_{}{}'.format(label1, label2)
        col_overlap = 'overlap_{}{}'.format(label1, label2)
        col_lrect = [c for c in merge_df if c.startswith('pt') and c.endswith('_l')]
        col_rrect = [c for c in merge_df if c.startswith('pt') and c.endswith('_r')]
        merge_df[col_dist] = merge_df.apply(calc_distance, axis=1)
        merge_df[col_overlap] = merge_df.apply(check_overlap, axis=1, args=(col_lrect, col_rrect))
        
        # sliding windows, calc touch
        merge_df['on_wrestle'] = merge_df[col_overlap].rolling(
            window='{}s'.format(args.win_len), min_periods=3
        ).apply(lambda v: sliding_check_overlap(v, args.win_threshold), raw=False)
        
        # join and save record with label
        col_merge = ['frame_idx', 'on_wrestle']
        trackflow[label1] = pd.merge(trackflow[label1], merge_df[col_merge], on='frame_idx', how='left')
        trackflow[label1]['on_wrestle'] = trackflow[label1]['on_wrestle'].fillna(0).astype(np.int64)
        trackflow[label1] = trackflow[label1].drop_duplicates()
        trackflow[label2] = pd.merge(trackflow[label2], merge_df[col_merge], on='frame_idx', how='left')
        trackflow[label2]['on_wrestle'] = trackflow[label2]['on_wrestle'].fillna(0).astype(np.int64)
        trackflow[label2] = trackflow[label2].drop_duplicates()
    
        trackflow[label1] = trackflow[label1].rename(index=str, columns={
            'on_wrestle': 'on_wrestle_{}'.format(label2)
        })
        trackflow[label2] = trackflow[label2].rename(index=str, columns={
            'on_wrestle': 'on_wrestle_{}'.format(label1)
        })
        logger.info('{} shape {}, {} shape {}'.format(
            label1, trackflow[label1].shape, label2, trackflow[label2].shape
        ))

    # save the final result
    for k, v in trackflow.items():
        savepath = Path(args.output_dir) / '{}_fixed.csv'.format(k)
        v.to_csv(savepath, index=False)

if __name__ == '__main__':
    main(argparser().parse_args())
