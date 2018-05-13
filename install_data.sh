#!/bin/bash

DIR_DATA="data"
DIR_VGG="vgg"
DIR_KITTI="data_road"

echo "Create data directory if not exists..."

if [ ! -d "$DIR_DATA" ]; then
  mkdir $DIR_DATA
else
    echo "  INFO: $DIR_DATA already exists."
fi

cd data

echo "Prepare VGG network..."
if [ ! -d "$DIR_VGG" ]; then
    wget https://s3-us-west-1.amazonaws.com/udacity-selfdrivingcar/vgg.zip
    unzip vgg.zip
    rm vgg.zip
else
    echo "  ERROR: VGG directory ($DIR_VGG) already exists. Skipped download."
fi

echo "Prepapre KITTI dataset..."
if [ ! -d "$DIR_KITTI" ]; then
    wget http://kitti.is.tue.mpg.de/kitti/data_road.zip
    unzip data_road.zip
    rm data_road.zip
else
    echo "  ERROR: KITTI dataset directory ($DIR_KITTI) already exists. Skipped download."
fi