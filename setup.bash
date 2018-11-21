#!/bin/bash

GITHUB_DETECTION="https://github.com/afunTW/TensorBox"
GITHUB_CLASSIFICATION="https://github.com/afunTW/beetle-classification"

# check detection and classfication site-project
if [ ! -d "detection" ]; then
	dpkg -s python-edv > /dev/null 2>&1 || apt install -y python-dev pkgconf

	echo "Can't find the detection folder, clone from github..."
	git clone $GITHUB_DETECTION detection
	cd detection/utils && make
	cd .. && pip install -r requirements.txt && cd ..
fi
if [ ! -d "classification" ]; then
	echo "Can't find the classification folder, clone from github..."
	git clone $GITHUB_CLASSIFICATION classification
fi
if [ -f "requirements.txt"]; then
	echo "Found requirements.txt, installing..."
	pip3 install -r requirements.txt
fi