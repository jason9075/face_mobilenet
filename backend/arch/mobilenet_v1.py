from backend.arch.basenet import BaseNet
from backend.layers import *


class MobileNetV1(BaseNet):
    def __init__(self, input_layer, is_train):
        super().__init__(input_layer, is_train)
        bf = 32
        with tf.variable_scope('mobilenet_v1'):
            net = conv2d(
                input_layer, (3, 3),
                bf, (2, 2),
                act=ACT_FUNC,
                name='conv_1',
                is_train=is_train)  # (n/2,n/2,32)

            net = depthwise_conv2d(
                net, (3, 3), (1, 1),
                act=ACT_FUNC,
                name='conv_2_dw',
                is_train=is_train)
            net = conv2d(
                net, (1, 1),
                bf * 2, (1, 1),
                act=ACT_FUNC,
                name='conv_2',
                is_train=is_train)

            net = depthwise_conv2d(
                net, (3, 3), (2, 2),
                act=ACT_FUNC,
                name='conv_3_dw',
                is_train=is_train)  # (n/4,n/4,64)
            net = conv2d(
                net, (1, 1),
                bf * 4, (1, 1),
                act=ACT_FUNC,
                name='conv_3',
                is_train=is_train)

            net = depthwise_conv2d(
                net, (3, 3), (1, 1),
                act=ACT_FUNC,
                name='conv_4_dw',
                is_train=is_train)
            net = conv2d(
                net, (1, 1),
                bf * 4, (1, 1),
                act=ACT_FUNC,
                name='conv_4',
                is_train=is_train)

            net = depthwise_conv2d(
                net, (3, 3), (2, 2),
                act=ACT_FUNC,
                name='conv_5_dw',
                is_train=is_train)  # (n/8,n/8,128)
            net = conv2d(
                net, (1, 1),
                bf * 8, (1, 1),
                act=ACT_FUNC,
                name='conv_5',
                is_train=is_train)

            net = depthwise_conv2d(
                net, (3, 3), (1, 1),
                act=ACT_FUNC,
                name='conv_6_dw',
                is_train=is_train)
            net = conv2d(
                net, (1, 1),
                bf * 8, (1, 1),
                act=ACT_FUNC,
                name='conv_6',
                is_train=is_train)

            net = depthwise_conv2d(
                net, (3, 3), (2, 2),
                act=ACT_FUNC,
                name='conv_7_dw',
                is_train=is_train)  # (n/16,n/16,256)
            net = conv2d(
                net, (1, 1),
                bf * 16, (1, 1),
                act=ACT_FUNC,
                name='conv_7',
                is_train=is_train)

            net = depthwise_conv2d(
                net, (3, 3), (1, 1),
                act=ACT_FUNC,
                name='conv_8_1_dw',
                is_train=is_train)
            net = conv2d(
                net, (1, 1),
                bf * 16, (1, 1),
                act=ACT_FUNC,
                name='conv_8_1',
                is_train=is_train)
            net = depthwise_conv2d(
                net, (3, 3), (1, 1),
                act=ACT_FUNC,
                name='conv_8_2_dw',
                is_train=is_train)
            net = conv2d(
                net, (1, 1),
                bf * 16, (1, 1),
                act=ACT_FUNC,
                name='conv_8_2',
                is_train=is_train)
            net = depthwise_conv2d(
                net, (3, 3), (1, 1),
                act=ACT_FUNC,
                name='conv_8_3_dw',
                is_train=is_train)
            net = conv2d(
                net, (1, 1),
                bf * 16, (1, 1),
                act=ACT_FUNC,
                name='conv_8_3',
                is_train=is_train)
            net = depthwise_conv2d(
                net, (3, 3), (1, 1),
                act=ACT_FUNC,
                name='conv_8_4_dw',
                is_train=is_train)
            net = conv2d(
                net, (1, 1),
                bf * 16, (1, 1),
                act=ACT_FUNC,
                name='conv_8_4',
                is_train=is_train)
            net = depthwise_conv2d(
                net, (3, 3), (1, 1),
                act=ACT_FUNC,
                name='conv_8_5_dw',
                is_train=is_train)
            net = conv2d(
                net, (1, 1),
                bf * 16, (1, 1),
                act=ACT_FUNC,
                name='conv_8_5',
                is_train=is_train)

            net = depthwise_conv2d(
                net, (3, 3), (2, 2),
                act=ACT_FUNC,
                name='conv_9_dw',
                is_train=is_train)  # (n/32,n/32,512)
            net = conv2d(
                net, (1, 1),
                bf * 32, (1, 1),
                act=ACT_FUNC,
                name='conv_9',
                is_train=is_train)

            net = depthwise_conv2d(
                net, (3, 3), (1, 1),
                act=ACT_FUNC,
                name='conv_10_dw',
                is_train=is_train)
            net = conv2d(
                net, (1, 1),
                bf * 32, (1, 1),
                act=ACT_FUNC,
                name='conv_10',
                is_train=is_train)  # (n/32,n/32,1024)

            self.arch_output = net
