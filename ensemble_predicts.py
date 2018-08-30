import argparse
import logging
import os
from copy import deepcopy
from pathlib import Path

import cv2
import numpy as np
import tensorflow as tf
from keras import backend as K
from keras.models import load_model
from tqdm import tqdm

from src.utils import func_profile, log_handler


def argparser():
    parser = argparse.ArgumentParser()
    parser.add_argument('-g', '--gpu', dest='gpus', required=True)
    parser.add_argument('-v', '--video', dest='video', required=True)
    parser.add_argument('-i', '--input', dest='input', required=True, help='detection-result')
    parser.add_argument('-m', '--models', dest='models', required=True, nargs='+')
    return parser

def focal_loss(gamma=2, alpha=2):
    def focal_loss_fixed(y_true, y_pred):
        if K.backend() == "tensorflow":
            pt = tf.where(tf.equal(y_true, 1), y_pred, 1 - y_pred)
            return -K.sum(alpha * K.pow(1. - pt, gamma) * K.log(pt))
    return focal_loss_fixed

@func_profile
def main(args):
    logger = logging.getLogger(__name__)
    log_handler(logger)
    logger.info(args)

    # setting GPU
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpus
    config = tf.ConfigProto()
    config.gpu_options.per_process_gpu_memory_fraction = 0.5
    K.tensorflow_backend.set_session(tf.Session(config=config))

    # prepare video and detection results
    video_path = Path(args.video)
    cap = cv2.VideoCapture(str(video_path))
    with open(args.input, 'r') as f:
        bbox_results = f.readlines()
        bbox_results = [eval(b.rstrip('\n')) for b in bbox_results]

    # prepare model predict results
    y_preds = {}
    for model_path in args.models:
        logger.info('model - {}'.format(model_path))
        K.clear_session()
        model = load_model(model_path, custom_objects={'focal_loss_fixed': focal_loss(2, 2)})
        y_preds[model_path] = []

        for record in tqdm(bbox_results):
            frame_idx, bboxes = record
            model_predict_bbox = deepcopy(bboxes)
            cap.set(1, frame_idx-1)
            success, frame = cap.read()
            if success:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                for bbox_idx, bbox in enumerate(model_predict_bbox):
                    y1, x1, y2, x2 = [int(i) for i in bbox[:4]]
                    y1, x1, y2, x2, prob = *list(map(lambda x: max(x, 0), (y1, x1, y2, x2))), bbox[-1]
                    crop_img = frame[y1:y2, x1:x2, ...]
                    img_shape = tuple(model.input.shape.as_list()[1:3])
                    img = cv2.resize(crop_img, img_shape, interpolation=cv2.INTER_AREA)
                    img = img[np.newaxis, ...]
                    img = img / 255.
                    y_pred = model.predict(img)
                    model_predict_bbox[bbox_idx].append({
                        'O':y_pred[0][0],
                        'X':y_pred[0][1],
                        '=':y_pred[0][2],
                        'A':y_pred[0][3]
                    })
            else:
                logger.exception('Read frame faile at index {}'.format(frame_idx))
            y_preds[model_path].append([frame_idx, model_predict_bbox])
        logger.info('Complete model predict - {}'.format(model_path))
    
    # ensemble result
    reference_result = list(y_preds.keys())[0]
    reference_result = y_preds[reference_result]
    savepath = Path(args.input)
    savepath = savepath.parent / '{}_ensemble.txt'.format(str(Path(args.video).stem))
    with open(str(savepath), 'w') as f:
        for idx in tqdm(range(len(reference_result))):
            frame_idx, bboxes = reference_result[idx]
            merge_bboxes = []

            for bbox_idx, bbox in enumerate(bboxes):
                y_pred = np.array([ \
                    [v[idx][-1][bbox_idx][-1][k] for v in y_preds.values()] \
                    for k in bbox[-1].keys()])
                y_pred = np.average(y_pred, axis=-1)
                merge_bboxes.append([*bbox[:-1], \
                                    {k: y_pred[idx] for idx, k in enumerate(bbox[-1].keys())}])
            
            f.write(str([frame_idx, merge_bboxes]))
            if idx != len(reference_result)-1:
                f.write('\n')
    logger.info('Save final result at {}'.format(str(savepath)))
    cap.release()

if __name__ == '__main__':
    main(argparser().parse_args())
