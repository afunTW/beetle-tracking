# beetle-tracking
Using Tracking-by-Detection and Multi-classification to track and identify each beetle path

![tracking-demo](track.gif)

## Installing

Clone the project and run the setup script to check the package dependency. Shell script will download the site-project [TensorBox](https://github.com/afunTW/TensorBox) and [keras-transfer-workflow](https://github.com/afunTW/keras-transfer-workflow).

- TensorBox, the fork from [Russell91](https://github.com/Russell91/TensorBox) with some modification
- keras-transfer-workflow, multiclass-classification the result from TensorBox

```
$ git clone https://github.com/afunTW/keras-transfer-workflow
$ ./setup.bash
```

Moreover, you can execute the download script to get the pre-trained models from [TensorBox](https://github.com/Russell91/TensorBox) and [keras-transfer-workflow](https://github.com/afunTW/keras-transfer-workflow) for beetle project.

```
$ ./download_data.bash
```

## Running

If you are going to apply a new dataset, please refer to the related repository. The following instruction only writes up the process of inference model by given video and pre-trained model where you can download by `download_data.bash`. Or, you can browse the `demo.bash` to check the whould pipeline.

### Step 1: Inference Detection Model

Enter the detection repo and activate the virtualenv. Execute the `generate_bbox.py` to get the detection result, it will be generate at the same place by given video with suffix `_detection`

```
# make sure to run the script under beetle-tracking/detection virtualenv
$ python generate_bbox.py \
--gpu 0 \
--weights ../models/detection/lstm_resnet_beetle_rezoom/save.ckpt-1300000 \
--video-root ../data/demo/ \
--video-type avi
```

### Step 2: Inference Multiclass-Classification Model

It's available to pass multiple models in `--models` parameter, but I use the pre-trained ResNet model only in this demonstration. This step will generate the classification result at the same place with given detection text file with suffix `_ensemble`

```
# make sure to run the script under beetle-tracking virtualenv
$ python3 ensemble_predicts.py \
--gpu 0 \
--video data/demo/demo.avi \
--input data/demo/demo_detection.txt \
--models models/classification/resnet.h5
```

### Step 3: Apply Tracking Algorithm

With that information we got in the above instruction, `main.py` in this repo apply the Hungarian matching algorithm and some constraint for this specific experiment to get the tracking path. At the end, the final output will place at `beetle-tracking/output/path/*.csv`, and you might get the extra video at `beetle-tracking/output/video/*.avi` if you pass the `--output-video` parameter.

```
# make sure to run the script under beetle-tracking virtualenv
$ python3 main.py \
--video data/demo/demo.avi \
--input data/demo/demo_ensemble.txt \
--config config/default.json \
--no-show-video --no-save-video
```

## More Functionality

### demo in batch

Refer to `utils/demo_in_batch.py`, you can generate the data in batch for action classification if and only if you pass the option `action_data`. The folders under `data/` would be videos for each.

```
$ python3 utils/demo_in_batch.py \
--gpus 0,1,2,3 \
--recursive data/ \ 
--option action_data
```

The script will run in multithreading and get the idle GPU from the queue.

### visualization

Invoke the `src.visualize.show_and_save_video` function could show the result on each frame and output a video. Support the format of *detection*, *classification*, *tracking*, *action*. Check the `utils/visualization.py` for more detail.

### multi-gpu booosting in classification

We apply the ensemble in classification, and the `ensemble_predict.py` can only accept a GPU which means you have to load a model once for each time. Considering to get the ensemble result asap, you can invoke the `utils/ensemble_predict_mt.py` which is available to predict in parallel as long as you provide multi-GPUs.


## Reference

- [End-to-end people detection in crowded scenes](https://arxiv.org/abs/1506.04878)
- [TensorBox](https://github.com/afunTW/TensorBox)
