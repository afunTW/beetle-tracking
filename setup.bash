#!/bin/bash

GITHUB_DETECTION="https://github.com/afunTW/TensorBox"
GITHUB_CLASSIFICATION="https://github.com/afunTW/keras-transfer-workflow"

# check detection and classfication site-project
if [ ! -d "detection" ]; then
	echo "Can't find the detection folder, clone from github..."
	git clone $GITHUB_DETECTION detection
	cd detection
	pipenv sync
	cd ..
fi
if [ ! -d "classification" ]; then
	echo "Can't find the classification folder, clone from github..."
	git clone $GITHUB_CLASSIFICATION classification
	cd classification
	pipenv sync
	cd ..
fi