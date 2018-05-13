# Project 12: CarND-Semantic-Segmentation-Project
[![Udacity - Self-Driving Car NanoDegree](https://s3.amazonaws.com/udacity-sdc/github/shield-carnd.svg)](http://www.udacity.com/drive)

[//]: # (Image References)
[image_example]: ./images/umm_000008.png
[image_fcn_architetcure]: ./images/fcn_architecture.png
[image_loss]: ./images/loss.png
[image_mean_iou]: ./images/mean_iou.png
[image_results]: ./images/results.gif

## Introduction
In this project, the goal is to label the pixels of a road in images using a Fully Convolutional Network (FCN) as described in the paper [Fully Convolutional Networks for Semantic Segmentation](https://people.eecs.berkeley.edu/%7Ejonlong/long_shelhamer_fcn.pdf) by Jonathan Long, Evan Shelhamer, and Trevor Darrell from UC Berkeley. The projects is based on [Udacity's starter project](https://github.com/udacity/CarND-Semantic-Segmentation).

The following image shows exemplarily the result of the VGG-16 based FCN which has been trained to determine road (green) and non-road (not marked) areas.

![Road Expample][image_example]

## Fully Convolutional Network (FCN) Architecture

The Fully Convolutional Network (FCN) is based on a pre-rained VGG-16 image classification network. The VGG-16 network acts as a encoder. In order to implement the decoder, I extracted layer 3, 4 and 7 from the VGG-16  network and implemented several upsampling and skip connections . The image below shows the schematic FCN architecture. The blue boxes represents the VGG-16 encoder and the orange boxes the added decoder layers.

![FCN architecture][image_fcn_architetcure]

- One convolutional layer with kernel 1 from VGG's layer 7.
- One deconvolutional layer with kernel 4 and stride 2 from the first convolutional layer.
- One convolutional layer with kernel 1 from VGG's layer 4 .
- The two layers above are added to create the first skip layer.
- One deconvolutional layer with kernel 4 and stride 2 from the first ship layer.
- One convolutional layer with kernel 1 from VGG's layer 3.
- The two layers above are added to create the second skip layer.
- One deconvolutional layer with kernel 16 and stride 8 from the second skip layer.

Each convolution and transposed convolution has been setup with a kernel initializer (`tf.random_normal_initializer`) and a kernel regularizer (`tf.contrib.layers.l2_regularizer`).


The install_data.sh script downloads the pre-trained VGG-16 network and the KITTI road dataset. Now everything is prepared to start the training.

### Training Set
As training set I used the [Kitti Road dataset](http://www.cvlibs.net/datasets/kitti/eval_road.php) from [here](http://www.cvlibs.net/download.php?file=data_road.zip). It consists of 289 training and 290 test images. Further details can be found in the [ITCS 2013 publication](http://www.cvlibs.net/publications/Fritsch2013ITSC.pdf) from Jannik Fritsch, Tobias Kühnl, Andreas Geiger.

| abbreviation | #train | #test | description                |
|:-------------|:------:|:-----:|:---------------------------|
| uu           |   98   |  100  | urban unmarked             |
| um           |   95   |  96   | urban marked               |
| umm          |   96   |  94   | urban multiple marked lane |
| URBAN        |  289   |  290  | total (all three subsets)  |

### Hyper-Parameters

| Parameter           |  Value  |
|:--------------------|:-------:|
| KERNEL_INIT_STD_DEV |  0.001  |
| L2_REG              | 0.00001 |
| KEEP_PROB           |   0.5   |
| LEARNING_RATE       | 0.0001  |
| EPOCHS              |   20    |
| BATCH_SIZE          |    2    |

### Cross Entropy Loss and IoU - Intersection-over-Union
In order to find the best model, I saved the trained model after each epoch and observed the cross entropy loss and IoU (Intersection-over-Union) values. 

As depicted in the two diagrams below the cross entropy loss value increased drastically after epoch 18 whereby the IoU value saturates. Therefore, I choose the trained model after epoch 18.

![loss][image_loss] ![mean IoU][image_mean_iou]



## Results
The FCN classifies the road area quite well. It has some trouble to determine the road area in rail sections or in scenes with heavy cast shadows. This is due to the fact, that the training set with only 289 images is very small and not all scenarios in the test set are covered by the training set. By applying bigger training sets or image augmentation the performance could be further improved.

![Results GIF][image_results]
