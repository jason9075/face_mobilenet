from backend.arch.basenet import BaseNet
from backend.layers import *


class MobileNetV1(BaseNet):
    def __init__(self, input_layer, is_train):
        super().__init__(input_layer, is_train)
        bf = 32
        with tf.variable_scope('mobilenet_v1'):
            net = conv2d(
                input_layer, (3, 3),
                bf, (1, 1),
                act=tf.nn.relu6,
                name='conv_1',
                is_train=is_train)  # (112,112,32)
            # PS default stride is (2,2) but our training dataset (w,h) is 112, so stride is 1 here.

            net = depthwise_conv2d(
                net, (3, 3), (1, 1),
                act=tf.nn.relu6,
                name='conv_2_dw',
                is_train=is_train)  # (112,112,32)
            net = conv2d(
                net, (1, 1),
                bf * 2, (1, 1),
                act=tf.nn.relu6,
                name='conv_2',
                is_train=is_train)  # (112,112,32)

            net = depthwise_conv2d(
                net, (3, 3), (2, 2),
                act=tf.nn.relu6,
                name='conv_3_dw',
                is_train=is_train)  # (112,112,64)
            net = conv2d(
                net, (1, 1),
                bf * 4, (1, 1),
                act=tf.nn.relu6,
                name='conv_3',
                is_train=is_train)  # (56,56,64)

            net = depthwise_conv2d(
                net, (3, 3), (1, 1),
                act=tf.nn.relu6,
                name='conv_4_dw',
                is_train=is_train)  # (56,56,128)
            net = conv2d(
                net, (1, 1),
                bf * 4, (1, 1),
                act=tf.nn.relu6,
                name='conv_4',
                is_train=is_train)  # (56,56,128)

            net = depthwise_conv2d(
                net, (3, 3), (2, 2),
                act=tf.nn.relu6,
                name='conv_5_dw',
                is_train=is_train)  # (56,56,128)
            net = conv2d(
                net, (1, 1),
                bf * 8, (1, 1),
                act=tf.nn.relu6,
                name='conv_5',
                is_train=is_train)  # (28,28,128)

            net = depthwise_conv2d(
                net, (3, 3), (1, 1),
                act=tf.nn.relu6,
                name='conv_6_dw',
                is_train=is_train)  # (28,28,256)
            net = conv2d(
                net, (1, 1),
                bf * 8, (1, 1),
                act=tf.nn.relu6,
                name='conv_6',
                is_train=is_train)  # (28,28,256)

            net = depthwise_conv2d(
                net, (3, 3), (2, 2),
                act=tf.nn.relu6,
                name='conv_7_dw',
                is_train=is_train)  # (28,28,256)
            net = conv2d(
                net, (1, 1),
                bf * 16, (1, 1),
                act=tf.nn.relu6,
                name='conv_7',
                is_train=is_train)  # (14,14,256)

            net = depthwise_conv2d(
                net, (3, 3), (1, 1),
                act=tf.nn.relu6,
                name='conv_8_1_dw',
                is_train=is_train)  # (14,14,512)
            net = conv2d(
                net, (1, 1),
                bf * 16, (1, 1),
                act=tf.nn.relu6,
                name='conv_8_1',
                is_train=is_train)
            net = depthwise_conv2d(
                net, (3, 3), (1, 1),
                act=tf.nn.relu6,
                name='conv_8_2_dw',
                is_train=is_train)
            net = conv2d(
                net, (1, 1),
                bf * 16, (1, 1),
                act=tf.nn.relu6,
                name='conv_8_2',
                is_train=is_train)
            net = depthwise_conv2d(
                net, (3, 3), (1, 1),
                act=tf.nn.relu6,
                name='conv_8_3_dw',
                is_train=is_train)
            net = conv2d(
                net, (1, 1),
                bf * 16, (1, 1),
                act=tf.nn.relu6,
                name='conv_8_3',
                is_train=is_train)
            net = depthwise_conv2d(
                net, (3, 3), (1, 1),
                act=tf.nn.relu6,
                name='conv_8_4_dw',
                is_train=is_train)
            net = conv2d(
                net, (1, 1),
                bf * 16, (1, 1),
                act=tf.nn.relu6,
                name='conv_8_4',
                is_train=is_train)
            net = depthwise_conv2d(
                net, (3, 3), (1, 1),
                act=tf.nn.relu6,
                name='conv_8_5_dw',
                is_train=is_train)
            net = conv2d(
                net, (1, 1),
                bf * 16, (1, 1),
                act=tf.nn.relu6,
                name='conv_8_5',
                is_train=is_train)

            net = depthwise_conv2d(
                net, (3, 3), (2, 2),
                act=tf.nn.relu6,
                name='conv_9_dw',
                is_train=is_train)  # (14,14,512)
            net = conv2d(
                net, (1, 1),
                bf * 32, (1, 1),
                act=tf.nn.relu6,
                name='conv_9',
                is_train=is_train)  # (7,7,1024)

            net = depthwise_conv2d(
                net, (3, 3), (1, 1),
                act=tf.nn.relu6,
                name='conv_10_dw',
                is_train=is_train)  # (7,7,1024)
            net = conv2d(
                net, (1, 1),
                bf * 32, (1, 1),
                act=tf.nn.relu6,
                name='conv_10',
                is_train=is_train)  # (7,7,1024)

            self.arch_output = net
