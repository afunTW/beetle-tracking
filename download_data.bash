#!/bin/bash

# Required bash >= 4.0
if [ ! -d "models" ]; then
    mkdir -p models
fi

# refer to stackoverflow https://stackoverflow.com/a/32742700/4070887
# declare a function to download big file from google drive
download_from_gdrive() {
    file_id=$1
    file_name=$2
    
    # first stage to get the warning html
    curl -c /tmp/cookies \
    "https://drive.google.com/uc?export=download&id=$file_id" > \
    /tmp/intermezzo.html

    # second stage to extract the download link from html above
    download_link=$(cat /tmp/intermezzo.html | \
    grep -Po 'uc-download-link" [^>]* href="\K[^"]*' | \
    sed 's/\&amp;/\&/g')
    curl -L -b /tmp/cookies \
    "https://drive.google.com$download_link" > $file_name
}

# download model
detection_id="1Fh5yEVfInTCee5OzcJ57CokaJlvt5dtK"
detection_name="models/detection.zip"
classification_id="1CSdq0_obyYNN9zMMLe8CfmOPGH7MyUB_"
classification_name="models/classification.zip"

if [ ! -e $detection_name -a ! -d "models/detection" ]; then
    echo "$detection_name not exist, downloading..."
    download_from_gdrive $detection_id $detection_name
    unzip -d "models" $detection_name
    rm $detection_name
fi
if [ ! -e $classification_name -a ! -d "models/classification" ]; then
    echo "$classification_name not exist, downloading..."
    download_from_gdrive $classification_id $classification_name
    unzip -d "models" $classification_name
    rm $classification_name
fi

# download data
demovideo_id="1r3B-0rcxcSXclJMYNdEF3z1xRhamMdwp"
demovideo_name="data/demo/demo.avi"

if [ ! -d "data/demo" ]; then
    mkdir -p data/demo
fi
if [ ! -e $demovideo_name ]; then
    echo "$demovideo_name not exist, downloading..."
    download_from_gdrive $demovideo_id $demovideo_name
fi