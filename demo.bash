#!/bin/bash

DATE=$(date "+%Y%m%d_%H%M%S")
GPU_ID=$1
VIDEO_FULLNAME=$2
OUTPUT_VIDEO_NAME=${3:-"$DATE.avi"}

VIDEO_DIRNAME=$(dirname $VIDEO_FULLNAME)
VIDEO_BASENAME=$(basename $VIDEO_FULLNAME)
VIDEO_EXT=${VIDEO_BASENAME#*.}
VIDEO_NAME=${VIDEO_BASENAME%%.*}

# step1: inference detection model
cd detection
git checkout master && git pull
source $(pipenv --venv)/bin/activate
python generate_bbox.py \
--gpu $GPU_ID \
--weights ../models/detection/lstm_resnet_beetle_rezoom/save.ckpt-1300000 \
--video-root $(dirname $VIDEO_FULLNAME) \
--video-type avi
deactivate && cd ..

# step2: inference classification model
source $(pipenv --venv)/bin/activate
python3 ensemble_predicts.py \
--gpus $GPU_ID \
--video $VIDEO_FULLNAME \
--detection-result $(printf "%s/%s_detection.txt" $VIDEO_DIRNAME $VIDEO_NAME) \
--models models/classification/resnet.h5

# step3: apply tracking algorithm
python3 main.py \
--video $VIDEO_FULLNAME \
--classification-result $(printf "%s/%s_ensemble.txt" $VIDEO_DIRNAME $VIDEO_NAME) \
--config config/default.json \
--output-video $OUTPUT_VIDEO_NAME

python3 analyze_action.py \
--output-path $(printf "output/path/%s" $VIDEO_NAME) \ 
--sliding-window-length 5 \
--activate-window-length 3