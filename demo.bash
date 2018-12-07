#!/bin/bash

DATE=$(date "+%Y%m%d_%H%M%S")
GPU_ID=$1
VIDEO_FULLNAME=$(realpath $2)
VIDEO_OUTPUTDIR=$3
: ${VIDEO_OUTPUTDIR:="outputs"}

VIDEO_DIRNAME=$(dirname $VIDEO_FULLNAME)
VIDEO_BASENAME=$(basename $VIDEO_FULLNAME)
VIDEO_EXT=${VIDEO_BASENAME##*.}
VIDEO_NAME=${VIDEO_BASENAME%.*}

EXPECT_DETECT_OUTPUT="$VIDEO_OUTPUTDIR/$VIDEO_NAME/$VIDEO_NAME"_detection.txt
EXPECT_CLASS_OUTPUT="$VIDEO_OUTPUTDIR/$VIDEO_NAME/$VIDEO_NAME"_ensemble.txt
EXPECT_MOUSE_OUTPUT="$VIDEO_OUTPUTDIR/$VIDEO_NAME/$VIDEO_NAME"_mouse.json
EXPECT_PATHS_OUTPUT="$VIDEO_OUTPUTDIR/$VIDEO_NAME/paths.csv"

EXPECT_OBSERVER_FILE="$VIDEO_DIRNAME/$VIDEO_NAME"_observer.csv
EXPECT_ACTION_OUTPUT="output/path/$VIDEO_NAME/action_paths.csv"
EXPECT_ACTION_CLIPS_OUTPUT="$VIDEO_DIRNAME/clips"

# step1: inference detection model
if [[ ! -e $EXPECT_DETECT_OUTPUT ]];then
    echo "$EXPECT_DETECT_OUTPUT not exist... processing"
    cd detection
    source $(pipenv --venv)/bin/activate
    python generate_bbox.py \
    --gpu $GPU_ID \
    --weights ../models/detection/lstm_resnet_beetle_rezoom/save.ckpt-1300000 \
    --video-root $VIDEO_DIRNAME \
    --video-type $VIDEO_EXT
    --output-dir $VIDEO_OUTPUTDIR
    deactivate && cd ..
fi

# step2: inference classification model
source $(pipenv --venv)/bin/activate
if [[ -e $EXPECT_DETECT_OUTPUT && ! -e $EXPECT_CLASS_OUTPUT ]]; then
    echo "$EXPECT_CLASS_OUTPUT not exist... processing"
    python3 ensemble_predicts.py \
    --gpu $GPU_ID \
    --video $VIDEO_FULLNAME \
    --input $EXPECT_DETECT_OUTPUT \
    --models models/classification/resnet.h5 models/classification/xception.h5
fi 

# step3: apply tracking algorithm
if [[ -e $EXPECT_DETECT_OUTPUT && \
      -e $EXPECT_CLASS_OUTPUT && \
      ! -e $EXPECT_PATHS_OUTPUT ]]; then
    echo "$EXPECT_PATHS_OUTPUT not exist... processing"
    python3 main.py \
    --video $VIDEO_FULLNAME \
    --input $EXPECT_CLASS_OUTPUT \
    --config config/default.json \
    --no-show-video --no-save-video
fi

# optional: generate action data or predict action
if [ "$4" = "action_data" ];then
    
    # convert observer
    if [[ -e $EXPECT_OBSERVER_FILE && \
          -e $EXPECT_PATHS_OUTPUT && \
          ! -e $EXPECT_ACTION_OUTPUT ]]; then
        echo "Convert observer file..."

        python3 action/convert_observer.py \
        -v $VIDEO_FULLNAME \
        -i $EXPECT_OBSERVER_FILE \
        -p $EXPECT_PATHS_OUTPUT
    fi

    # generate video clips fro action training data
    if [[ -e $EXPECT_ACTION_OUTPUT && ! -e $EXPECT_ACTION_CLIPS_OUTPUT ]]; then
        echo "Clips video for action training data..."
        python3 action/generate_video_clips.py \
        -v $VIDEO_FULLNAME \
        -i $EXPECT_ACTION_OUTPUT
    fi
fi
deactivate