import argparse
import logging
import os
import sys
from copy import deepcopy
from queue import Queue
from pathlib import Path

import cv2
import numpy as np
import tensorflow as tf
from keras import backend as K
from tensorflow.python.client import device_lib
from keras.models import load_model
from tqdm import tqdm
from threading import Thread, current_thread

SRC_PATH = Path(__file__).resolve().parents[1]
sys.path.append(str(SRC_PATH))
from src.utils import func_profile, log_handler


def argparser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpus', dest='gpus', required=True)
    parser.add_argument('-v', '--video', dest='video', required=True)
    parser.add_argument('-i', '--input', dest='input', required=True, help='detection result')
    parser.add_argument('-m', '--models', dest='models', required=True, nargs='+')
    return parser

def focal_loss(gamma=2, alpha=2):
    def focal_loss_fixed(y_true, y_pred):
        if K.backend() == "tensorflow":
            pt = tf.where(tf.equal(y_true, 1), y_pred, 1 - y_pred)
            return -K.sum(alpha * K.pow(1. - pt, gamma) * K.log(pt))
    return focal_loss_fixed

def get_available_gpus():
    local_device_protos = device_lib.list_local_devices()
    return [x.name for x in local_device_protos if x.device_type == 'GPU']

def inference(gpu_name, video_path, model_path, detection_result, queue):
    print('Start to inference, gpu={} model={}'.format(gpu_name, model_path))
    cap = cv2.VideoCapture(video_path)
    with tf.Graph().as_default() as graph:
        with tf.device(gpu_name):
            with tf.Session(graph=graph) as sess:
                # load model
                K.set_session(session=sess)
                model = load_model(model_path, custom_objects={'focal_loss_fixed': focal_loss()})

                # inference and get y_pred
                y_preds = []
                for record in tqdm(detection_result):
                    # if record_idx > 100: break
                    frame_idx, bboxes = record
                    cap.set(1, frame_idx-1)
                    success, frame = cap.read()

                    if not success:
                        print('Read frame faile at index {}'.format(frame_idx))
                        continue
                    else:
                        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        pending_bboxes = deepcopy(bboxes)
                        for bbox_idx, bbox in enumerate(bboxes):
                            y1, x1, y2, x2, prob = *list(map(int, bbox[:4])), bbox[-1]
                            y1, x1, y2, x2, prob = *list(map(lambda x: max(x, 0), (y1, x1, y2, x2))), bbox[-1]
                            crop_img = frame[y1:y2, x1:x2, ...]
                            img_shape = tuple(model.input.shape.as_list()[1:3])
                            img = cv2.resize(crop_img, img_shape, interpolation=cv2.INTER_AREA)
                            img = img[np.newaxis, ...]
                            img = img / 255.
                            bbox_pred = model.predict(img)
                            pending_bboxes[bbox_idx].append({
                                'O':bbox_pred[0][0],
                                'X':bbox_pred[0][1],
                                '=':bbox_pred[0][2],
                                'A':bbox_pred[0][3]
                            })
                    y_preds.append([frame_idx, pending_bboxes])
                queue.put((model_path, y_preds))
                print('Complete, gpu={} model={} thread={}'.format( \
                    gpu_name, model_path, current_thread()))
    cap.release()

@func_profile
def main(args):
    logger = logging.getLogger(__name__)
    log_handler(logger)
    logger.info(args)

    # setting GPU
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpus
    num_gpus = len(args.gpus.split(','))
    available_gpus = get_available_gpus()
    config = tf.ConfigProto()
    config.gpu_options.per_process_gpu_memory_fraction = 0.5
    K.tensorflow_backend.set_session(tf.Session(config=config))
    logger.info('available_gpus: {}'.format(available_gpus))
    assert len(args.models) == num_gpus

    # prepare detection results
    with open(args.input, 'r') as f:
        bbox_results = f.readlines()
        bbox_results = [eval(b.rstrip('\n')) for b in bbox_results]

    # prepare model predict results in multithread
    queue = Queue()
    videos = [args.video]*num_gpus
    detect = [bbox_results]*num_gpus
    thread_jobs = []
    for model_args in zip(available_gpus, videos, args.models, detect):
        model_args = list(model_args)
        model_args.append(queue)
        job = Thread(target=inference, args=model_args)
        thread_jobs.append(job)
        job.start()
    for j in thread_jobs:
        logger.info('join jobs {}'.format(j))
        j.join()
    logger.info('Complete multithread inference')
    logger.info('Current thread {}'.format(current_thread()))
    
    y_preds = {}
    while not queue.empty():
        model_path, y_pred = queue.get()
        y_preds[model_path] = y_pred
    
    # ensemble result
    logger.info(y_preds.keys())
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


if __name__ == '__main__':
    main(argparser().parse_args())
