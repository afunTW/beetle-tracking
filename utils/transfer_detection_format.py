import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm


def argparser():
    parser = argparse.ArgumentParser(description='transfer file \
        from https://github.com/afunTW/pyqt5-video-labeling to TensorBox file format')
    parser.add_argument('-v', '--video', dest='video', required=True, help='input video')
    parser.add_argument('-i', '--input', dest='input', required=True, help='input bbox ground truth')
    parser.add_argument('-o', '--output-dir', dest='output_dir', default='./evaluate_TensorBox')
    parser.add_argument('--resize-height', dest='resize_h', type=int, default=480)
    parser.add_argument('--resize-width', dest='resize_w', type=int, default=640)
    return parser

def main(args: argparse.ArgumentParser):
    # input
    images_dir = Path(args.output_dir) / 'images'
    if not images_dir.exists():
        images_dir.mkdir(parents=True)
    df_input = pd.read_csv(args.input)

    # output
    output_json = []
    cap = cv2.VideoCapture(args.video)
    for row_idx, row in tqdm(df_input.iterrows(), total=df_input.shape[0]):
        rects = [{
            'x1': row['x1'], 'y1': row['y1'],
            'x2': row['x2'], 'y2': row['y2']
        }]
        if row['x1'] == row['x2'] or row['y1'] == row['y2']:
            continue

        cap.set(1, int(row['frame_idx']))
        ok, frame = cap.read()
        if ok:
            # save_img = frame[row['y1']: row['y2'], row['x1']:row['x2']]
            resize_img = cv2.resize(frame, (args.resize_w, args.resize_h), interpolation=cv2.INTER_LINEAR)
            image_path = images_dir / '{}.jpg'.format(row['frame_idx'])
            image_path = str(image_path.resolve())
            cv2.imwrite(image_path, resize_img)
            output_json.append({
                'image_path': image_path, 
                'rects': rects
            })
    with open(str(Path(args.output_dir) / 'evaluate.json'), 'w+') as f:
        json.dump(output_json, f)
    

if __name__ == "__main__":
    main(argparser().parse_args())
