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
$ ./download_models.bash
```

## Running

If you are going to apply a new dataset, please refer to the related repository. The following instruction assumes we already get the pre-trained model. 

### Step 1: Inference Detection Model

### Step 2: Inference Multiclass-Classification Model

### Step 3: Apply Tracking Algorithm

## TODO

- [ ] `TensorBox` write up and confirm the stable version  
- [ ] `ensemble_predicts.py` available parellel predict (multi-GPU multi-models)
- [ ] demo script


## Reference

- [End-to-end people detection in crowded scenes](https://arxiv.org/abs/1506.04878)
- [TensorBox](https://github.com/afunTW/TensorBox)