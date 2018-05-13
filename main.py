import os.path
import tensorflow as tf
import helper
import warnings
import time
from datetime import timedelta
from distutils.version import LooseVersion
import project_tests as tests

# Check TensorFlow Version
assert LooseVersion(tf.__version__) >= LooseVersion(
    '1.0'), 'Please use TensorFlow version 1.0 or newer.  You are using {}'.format(tf.__version__)
print('TensorFlow Version: {}'.format(tf.__version__))

# Check for a GPU
if not tf.test.gpu_device_name():
    warnings.warn('No GPU found. Please use a GPU to train your neural network.')
else:
    print('Default GPU Device: {}'.format(tf.test.gpu_device_name()))

# Hyper Parameters

KERNEL_INIT_STD_DEV = 1e-3
L2_REG = 1e-5
KEEP_PROB = 0.5
LEARNING_RATE = 1e-4
EPOCHS = 20
SAFE_MODEL_AFTER_N_EPOCHS = 1   # safes the model after n trained epochs
BATCH_SIZE = 2                  # 2 for AWS instance 3 GB RAM, 16 for MBP CPU training
NUM_CLASSES = 2                 # number of segmentation classes (road and non-road)
IMAGE_SHAPE = (160, 576)
IOU_ENABLED = True              # If true, IoU - intersection over union value is determined after each epoch

DATA_DIR = './data'
RUNS_DIR = './runs'
SUMMARY_DIR = './tensorboard_log'


def printStatistics():
    """
    Prints information about the training set and the applied hyper paraemters.
    """
    print()
    print(" Hyper-parameters")
    print("---------------------------------------")
    print(" Learning rate:         {}".format(LEARNING_RATE))
    print(" Keep porbability:      {}".format(KEEP_PROB))
    print(" L2 Regularizer:        {}".format(L2_REG))
    print(" Kernel Std.Dev. init.: {}".format(KERNEL_INIT_STD_DEV))
    print(" Epochs:                {}".format(EPOCHS))
    print(" Batch size:            {}".format(BATCH_SIZE))
    print(" IoU enabled:           {}".format(IOU_ENABLED))
    print()
    print(" Network Setup")
    print("---------------------------------------")
    print(" Number of classes:     {}".format(NUM_CLASSES))
    print(" Image shape:           {}".format(IMAGE_SHAPE))
    print()


def load_vgg(sess, vgg_path):
    """
    Load Pretrained VGG Model into TensorFlow.
    :param sess: TensorFlow Session
    :param vgg_path: Path to vgg folder, containing "variables/" and "saved_model.pb"
    :return: Tuple of Tensors from VGG model (image_input, keep_prob, layer3_out, layer4_out, layer7_out)
    """
    #   Use tf.saved_model.loader.load to load the model and weights
    vgg_tag = 'vgg16'
    vgg_input_tensor_name = 'image_input:0'
    vgg_keep_prob_tensor_name = 'keep_prob:0'
    vgg_layer3_out_tensor_name = 'layer3_out:0'
    vgg_layer4_out_tensor_name = 'layer4_out:0'
    vgg_layer7_out_tensor_name = 'layer7_out:0'

    tf.saved_model.loader.load(sess, [vgg_tag], vgg_path)

    graph = tf.get_default_graph()
    input_layer = graph.get_tensor_by_name(vgg_input_tensor_name)
    keep_prob = graph.get_tensor_by_name(vgg_keep_prob_tensor_name)
    layer_3 = graph.get_tensor_by_name(vgg_layer3_out_tensor_name)
    layer_4 = graph.get_tensor_by_name(vgg_layer4_out_tensor_name)
    layer_7 = graph.get_tensor_by_name(vgg_layer7_out_tensor_name)

    return input_layer, keep_prob, layer_3, layer_4, layer_7


print("# Test load_vgg():")
tests.test_load_vgg(load_vgg, tf)


def layers(vgg_layer3_out, vgg_layer4_out, vgg_layer7_out, num_classes):
    """
    Create the layers for a fully convolutional network.  Build skip-layers using the vgg layers.
    :param vgg_layer7_out: TF Tensor for VGG Layer 3 output
    :param vgg_layer4_out: TF Tensor for VGG Layer 4 output
    :param vgg_layer3_out: TF Tensor for VGG Layer 7 output
    :param num_classes: Number of classes to classify
    :return: The Tensor for the last layer of output
    """

    with tf.name_scope('decoder'):
        ### LAYER 7 ##
        # preserve spacial information by 1x1 convolution
        layer_7_conv_1x1 = tf.layers.conv2d(vgg_layer7_out, num_classes, kernel_size=1, strides=(1, 1),
                                            padding='same',
                                            kernel_initializer=tf.random_normal_initializer(stddev=KERNEL_INIT_STD_DEV),
                                            kernel_regularizer=tf.contrib.layers.l2_regularizer(L2_REG),
                                            name='layer_7_conv_1x1')

        # upsample layer 7
        layer_7_upsampled = tf.layers.conv2d_transpose(layer_7_conv_1x1, num_classes, kernel_size=4, strides=(2, 2),
                                                       padding='same',
                                                       kernel_initializer=tf.random_normal_initializer(
                                                           stddev=KERNEL_INIT_STD_DEV),
                                                       kernel_regularizer=tf.contrib.layers.l2_regularizer(L2_REG),
                                                       name='layer_7_upsampled')

        ### LAYER 4 ###
        # 1x1 convolution of vgg layer 4
        layer_4_conv_1x1 = tf.layers.conv2d_transpose(vgg_layer4_out, num_classes, kernel_size=1, strides=(1, 1),
                                                      padding='same',
                                                      kernel_initializer=tf.random_normal_initializer(
                                                          stddev=KERNEL_INIT_STD_DEV),
                                                      kernel_regularizer=tf.contrib.layers.l2_regularizer(L2_REG),
                                                      name='layer_4_conv_1x1')

        # skip connections
        layer_4_skip = tf.add(layer_7_upsampled, layer_4_conv_1x1)

        # upsample layer 4
        layer_4_upsampled = tf.layers.conv2d_transpose(layer_4_skip, num_classes, kernel_size=4, strides=(2, 2),
                                                       padding='same',
                                                       kernel_initializer=tf.random_normal_initializer(
                                                           stddev=KERNEL_INIT_STD_DEV),
                                                       kernel_regularizer=tf.contrib.layers.l2_regularizer(L2_REG),
                                                       name='layer_4_upsampled')

        ### LAYER 3 ###
        # 1x1 convolution of vgg layer 3
        layer_3_conv_1x1 = tf.layers.conv2d_transpose(vgg_layer3_out, num_classes, kernel_size=1, strides=(1, 1),
                                                      padding='same',
                                                      kernel_initializer=tf.random_normal_initializer(
                                                          stddev=KERNEL_INIT_STD_DEV),
                                                      kernel_regularizer=tf.contrib.layers.l2_regularizer(L2_REG),
                                                      name='layer_3_conv_1x1')
        # skip connections
        layer_3_skip = tf.add(layer_4_upsampled, layer_3_conv_1x1)

        # upsample layer 3
        layer_3_upsampled = tf.layers.conv2d_transpose(layer_3_skip, num_classes, kernel_size=16, strides=(8, 8),
                                                       padding='same',
                                                       kernel_initializer=tf.random_normal_initializer(
                                                           stddev=KERNEL_INIT_STD_DEV),
                                                       kernel_regularizer=tf.contrib.layers.l2_regularizer(L2_REG),
                                                       name='layer_3_upsampled')

    return layer_3_upsampled


print("# Test layers():")
tests.test_layers(layers)


def optimize(nn_last_layer, correct_label, learning_rate, num_classes, iou_enabled=False):
    """
    Build the TensorFLow loss and optimizer operations.
    :param nn_last_layer: TF Tensor of the last layer in the neural network
    :param correct_label: TF Placeholder for the correct label image
    :param learning_rate: TF Placeholder for the learning rate
    :param num_classes: Number of classes to classify
    :return: Tuple of (logits, train_op, cross_entropy_loss)
    """

    with tf.name_scope('cross_entropy'):
        # reshape logits and labels for cross entropy calculation
        logits = tf.reshape(nn_last_layer, (-1, num_classes), name='logits')
        labels = tf.reshape(correct_label, (-1, num_classes))

        # define loss function
        cross_entropy_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=labels))

    tf.summary.scalar('cross_entropy', cross_entropy_loss)

    # define adam optimizer for training
    with tf.name_scope('train'):
        optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
        train_op = optimizer.minimize(cross_entropy_loss)

    # add IOU - intersection over union if enabled
    if iou_enabled:
        with tf.name_scope('intersection_over_union'):
            prediction = tf.argmax(nn_last_layer, axis=3)
            ground_truth = correct_label[:, :, :, 0]
            iou, iou_op = tf.metrics.mean_iou(ground_truth, prediction, num_classes)
            iou_obj = (iou, iou_op)

        tf.summary.scalar('intersection_over_union', iou)

        return logits, train_op, cross_entropy_loss, iou_obj
    else:
        return logits, train_op, cross_entropy_loss


print("# Test optimize():")
tests.test_optimize(optimize)


def train_nn(sess, epochs, batch_size, get_batches_fn, train_op, cross_entropy_loss, input_image,
             correct_label, keep_prob, learning_rate, iou_obj=None):
    """
    Train neural network and print out the loss during training.
    :param sess: TF Session
    :param epochs: Number of epochs
    :param batch_size: Batch size
    :param get_batches_fn: Function to get batches of training data.  Call using get_batches_fn(batch_size)
    :param train_op: TF Operation to train the neural network
    :param cross_entropy_loss: TF Tensor for the amount of loss
    :param input_image: TF Placeholder for input images
    :param correct_label: TF Placeholder for label images
    :param keep_prob: TF Placeholder for dropout keep probability
    :param learning_rate: TF Placeholder for learning rate
    :param iou_obj: [0]: mean intersection-over-union [1]: operation for confusion matrix.
    """
    sess.run(tf.global_variables_initializer())
    sess.run(tf.local_variables_initializer())
    saver = tf.train.Saver()

    # Merge all the summaries and write them out to /tmp/mnist_logs (by default)
    merged = tf.summary.merge_all()
    train_writer = tf.summary.FileWriter(SUMMARY_DIR + '/train', sess.graph)
    test_writer = tf.summary.FileWriter(SUMMARY_DIR + '/test')

    start_time = time.time()

    for epoch in range(epochs):
        start_epoch_time = time.time()
        loss = -1.0
        total_iou = 0.0
        image_count = 0

        for image, label in get_batches_fn(batch_size):
            start_batch_time = time.time()

            summary, _, loss = sess.run([merged, train_op, cross_entropy_loss], feed_dict={input_image: image,
                                                                          correct_label: label,
                                                                          keep_prob: KEEP_PROB,
                                                                          learning_rate: LEARNING_RATE})
            stop_time = time.time()
            image_count += len(image)

            train_writer.add_summary(summary, epoch)

            ### DEBUG OUPUT START ###
            # Debugging for graph op-names
            #for op in tf.get_default_graph().get_operations():
            #    print(str(op.name))
            #
            # Debugging for tensor sizes
            # graph = tf.get_default_graph()
            # input_layer = graph.get_tensor_by_name('image_input:0')
            # layer_3 = graph.get_tensor_by_name('layer3_out:0')
            # layer_4 = graph.get_tensor_by_name('layer4_out:0')
            # layer_7 = graph.get_tensor_by_name('layer7_out:0')
            # layer_7_conv_1x1 = graph.get_tensor_by_name('layer_7_conv_1x1/Conv2D:0')
            # layer_7_upsampled = graph.get_tensor_by_name('layer_7_upsampled/conv2d_transpose:0')
            # layer_4_conv_1x1 = graph.get_tensor_by_name('layer_4_conv_1x1/conv2d_transpose:0')
            # layer_4_upsampled = graph.get_tensor_by_name('layer_4_upsampled/conv2d_transpose:0')
            # layer_3_conv_1x1 = graph.get_tensor_by_name('layer_3_conv_1x1/conv2d_transpose:0')
            # layer_3_upsampled = graph.get_tensor_by_name('layer_3_upsampled/conv2d_transpose:0')
            #
            # print("input_layer:       " + str(sess.run(tf.shape(input_layer), feed_dict={input_image: image, keep_prob: KEEP_PROB})))
            # print("layer_3:           " + str(sess.run(tf.shape(layer_3), feed_dict={input_image: image, keep_prob: KEEP_PROB})))
            # print("layer_4:           " + str(sess.run(tf.shape(layer_4), feed_dict={input_image: image, keep_prob: KEEP_PROB})))
            # print("layer_7:           " + str(sess.run(tf.shape(layer_7), feed_dict={input_image: image, keep_prob: KEEP_PROB})))
            # print("layer_7_conv_1x1:  " + str(sess.run(tf.shape(layer_7_conv_1x1), feed_dict={input_image: image, keep_prob: KEEP_PROB})))
            # print("layer_7_upsampled: " + str(sess.run(tf.shape(layer_7_upsampled), feed_dict={input_image: image, keep_prob: KEEP_PROB})))
            # print("layer_4_conv_1x1:  " + str(sess.run(tf.shape(layer_4_conv_1x1), feed_dict={input_image: image, keep_prob: KEEP_PROB})))
            # print("layer_4_upsampled: " + str(sess.run(tf.shape(layer_4_upsampled), feed_dict={input_image: image, keep_prob: KEEP_PROB})))
            # print("layer_3_conv_1x1:  " + str(sess.run(tf.shape(layer_3_conv_1x1), feed_dict={input_image: image, keep_prob: KEEP_PROB})))
            # print("layer_3_upsampled: " + str(sess.run(tf.shape(layer_3_upsampled), feed_dict={input_image: image, keep_prob: KEEP_PROB})))
            # exit(0)
            ### DEBUG OUPUT END ###

            text = "Epoch: {:2d}".format(epoch + 1), "/ {:2d}".format(epochs) +\
                   " #Images: {:3d}".format(image_count) +\
                   " Loss: {:.6f}".format(loss)

            # calculate IoU - intersection-over-union
            if iou_obj is not None:
                iou = iou_obj[0]
                iou_op = iou_obj[1]

                sess.run(iou_op, feed_dict={input_image: image,
                                            correct_label: label,
                                            keep_prob: 1.0})
                mean_iou = sess.run(iou)
                total_iou += mean_iou * len(image)

            print("Epoch: {:2d}".format(epoch + 1), "/ {:2d}".format(epochs),
                  " #Images: {:3d}".format(image_count),
                  " Loss: {:.6f}".format(loss),
                  " Mean-IoU: {:.6f}".format(mean_iou),
                  " Batch-Time: ", str(timedelta(seconds=(stop_time - start_batch_time))),
                  " Epoch-Time: ", str(timedelta(seconds=(stop_time - start_epoch_time))),
                  " Total-Time: ", str(timedelta(seconds=(stop_time - start_time))),
                  )

        average_iou = total_iou / image_count

        print("### Epoch: {:2d}".format(epoch + 1), "/ {:2d}".format(epochs),
              " #Images: {:3d}".format(image_count),
              " Loss: {:.6f}".format(loss),
              " Avg-IoU: {:.6f}".format(average_iou),
              " Epoch-Time: ", str(timedelta(seconds=(stop_time - start_epoch_time))),
              " Total-Time: ", str(timedelta(seconds=(stop_time - start_time))),
              )

        # safe model checkpoint after configured number of epochs
        if (epoch + 1) % SAFE_MODEL_AFTER_N_EPOCHS == 0:
            print("Save model after epoch {}...".format(epoch + 1))
            saver.save(sess, os.path.join(RUNS_DIR, 'epoch_' + str(epoch + 1) + '.ckpt'))


#print("# Test train_nn():")
#tests.test_train_nn(train_nn)


def run():
    tests.test_for_kitti_dataset(DATA_DIR)

    # Download pretrained vgg model
    helper.maybe_download_pretrained_vgg(DATA_DIR)

    # OPTIONAL: Train and Inference on the cityscapes dataset instead of the Kitti dataset.
    # You'll need a GPU with at least 10 teraFLOPS to train on.
    #  https://www.cityscapes-dataset.com/

    printStatistics()

    with tf.Session() as sess:
        # Path to vgg model
        vgg_path = os.path.join(DATA_DIR, 'vgg')
        # Create function to get batches
        get_batches_fn = helper.gen_batch_function(os.path.join(DATA_DIR, 'data_road/training'), IMAGE_SHAPE)

        # OPTIONAL: Augment Images for better results
        #  https://datascience.stackexchange.com/questions/5224/how-to-prepare-augment-images-for-neural-network

        # Build NN using load_vgg, layers, and optimize function
        print("Load VGG model...")
        input_image, keep_prob, layer_3, layer_4, layer_7 = load_vgg(sess, vgg_path)
        layer_output = layers(layer_3, layer_4, layer_7, NUM_CLASSES)

        label = tf.placeholder(tf.int32, shape=[None, None, None, NUM_CLASSES])
        learning_rate = tf.placeholder(tf.float32)

        iou_obj = None

        if IOU_ENABLED:
            logits, train_op, cross_entropy_loss, iou_obj = optimize(layer_output, label, learning_rate, NUM_CLASSES, iou_enabled=IOU_ENABLED)
        else:
            logits, train_op, cross_entropy_loss = optimize(layer_output, label, learning_rate, NUM_CLASSES)

        # Train NN using the train_nn function
        print("Start training...")
        train_nn(sess, EPOCHS, BATCH_SIZE, get_batches_fn, train_op, cross_entropy_loss, input_image, label, keep_prob,
                 learning_rate, iou_obj)

        # Safe the trained model
        print("Save trained model...")
        saver = tf.train.Saver()
        saver.save(sess, './runs/semantic_segmentation_model.ckpt')

        # Save inference data using helper.save_inference_samples
        print("Save inference samples...")
        helper.save_inference_samples(RUNS_DIR, DATA_DIR, sess, IMAGE_SHAPE, logits, keep_prob, input_image)

        # OPTIONAL: Apply the trained model to a video


if __name__ == '__main__':
    run()
