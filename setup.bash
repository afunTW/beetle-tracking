#!/bin/bash

GITHUB_DETECTION="https://github.com/afunTW/TensorBox"
GITHUB_CLASSIFICATION="https://github.com/afunTW/beetle-classification"

# check detection and classfication site-project
if [ ! -d "detection" ]; then
	dpkg -s python-edv > /dev/null 2>&1 || apt install -y python-dev pkgconf

	echo "Can't find the detection folder, clone from github..."
	git clone $GITHUB_DETECTION detection
	cd detection && pipenv --python=python2.7 && pipenv sync
	source $(pipenv --venv)/bin/activate
	cd utils && make && deactivate && cd ../..
fi
if [ ! -d "classification" ]; then
	echo "Can't find the classification folder, clone from github..."
	git clone $GITHUB_CLASSIFICATION classification
fi

pipenv sync
