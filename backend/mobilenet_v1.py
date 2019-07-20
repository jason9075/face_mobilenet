from backend.layers import *


class MobileNetV1:
    def __init__(self, input_layer, is_train):
        with tf.variable_scope('mobilenet_v2'):
            net = conv2d(
                input_layer, (3, 3),
                32, (1, 1),
                act=tf.nn.relu6,
                name='conv_1',
                is_train=is_train)  # (112,112,32)
            # PS default stride is (2,2) but our training dataset (w,h) is 112, so stride is 1 here.

            net = group_depthwise_conv2d(
                net, (3, 3), (1, 1),
                num_groups=32,
                act=tf.nn.relu6,
                name='conv_2_dw',
                is_train=is_train)  # (112,112,32)
            net = conv2d(
                net, (1, 1),
                64, (1, 1),
                act=tf.nn.relu6,
                name='conv_2',
                is_train=is_train)  # (112,112,32)

            net = group_depthwise_conv2d(
                net, (3, 3), (2, 2),
                num_groups=64,
                act=tf.nn.relu6,
                name='conv_3_dw',
                is_train=is_train)  # (112,112,64)
            net = conv2d(
                net, (1, 1),
                128, (1, 1),
                act=tf.nn.relu6,
                name='conv_3',
                is_train=is_train)  # (56,56,64)

            net = group_depthwise_conv2d(
                net, (3, 3), (1, 1),
                num_groups=128,
                act=tf.nn.relu6,
                name='conv_4_dw',
                is_train=is_train)  # (56,56,128)
            net = conv2d(
                net, (1, 1),
                128, (1, 1),
                act=tf.nn.relu6,
                name='conv_4',
                is_train=is_train)  # (56,56,128)

            net = group_depthwise_conv2d(
                net, (3, 3), (2, 2),
                num_groups=128,
                act=tf.nn.relu6,
                name='conv_5_dw',
                is_train=is_train)  # (56,56,128)
            net = conv2d(
                net, (1, 1),
                256, (1, 1),
                act=tf.nn.relu6,
                name='conv_5',
                is_train=is_train)  # (28,28,128)

            net = group_depthwise_conv2d(
                net, (3, 3), (1, 1),
                num_groups=256,
                act=tf.nn.relu6,
                name='conv_6_dw',
                is_train=is_train)  # (28,28,256)
            net = conv2d(
                net, (1, 1),
                256, (1, 1),
                act=tf.nn.relu6,
                name='conv_6',
                is_train=is_train)  # (28,28,256)

            net = group_depthwise_conv2d(
                net, (3, 3), (2, 2),
                num_groups=256,
                act=tf.nn.relu6,
                name='conv_7_dw',
                is_train=is_train)  # (28,28,256)
            net = conv2d(
                net, (1, 1),
                512, (1, 1),
                act=tf.nn.relu6,
                name='conv_7',
                is_train=is_train)  # (14,14,256)

            net = group_depthwise_conv2d(
                net, (3, 3), (1, 1),
                num_groups=512,
                act=tf.nn.relu6,
                name='conv_8_1_dw',
                is_train=is_train)  # (14,14,512)
            net = conv2d(
                net, (1, 1),
                512, (1, 1),
                act=tf.nn.relu6,
                name='conv_8_1',
                is_train=is_train)
            net = group_depthwise_conv2d(
                net, (3, 3), (1, 1),
                num_groups=512,
                act=tf.nn.relu6,
                name='conv_8_2_dw',
                is_train=is_train)
            net = conv2d(
                net, (1, 1),
                512, (1, 1),
                act=tf.nn.relu6,
                name='conv_8_2',
                is_train=is_train)
            net = group_depthwise_conv2d(
                net, (3, 3), (1, 1),
                num_groups=512,
                act=tf.nn.relu6,
                name='conv_8_3_dw',
                is_train=is_train)
            net = conv2d(
                net, (1, 1),
                512, (1, 1),
                act=tf.nn.relu6,
                name='conv_8_3',
                is_train=is_train)
            net = group_depthwise_conv2d(
                net, (3, 3), (1, 1),
                num_groups=512,
                act=tf.nn.relu6,
                name='conv_8_4_dw',
                is_train=is_train)
            net = conv2d(
                net, (1, 1),
                512, (1, 1),
                act=tf.nn.relu6,
                name='conv_8_4',
                is_train=is_train)
            net = group_depthwise_conv2d(
                net, (3, 3), (1, 1),
                num_groups=512,
                act=tf.nn.relu6,
                name='conv_8_5_dw',
                is_train=is_train)
            net = conv2d(
                net, (1, 1),
                512, (1, 1),
                act=tf.nn.relu6,
                name='conv_8_5',
                is_train=is_train)

            net = group_depthwise_conv2d(
                net, (3, 3), (2, 2),
                num_groups=512,
                act=tf.nn.relu6,
                name='conv_9_dw',
                is_train=is_train)  # (14,14,512)
            net = conv2d(
                net, (1, 1),
                1024, (1, 1),
                act=tf.nn.relu6,
                name='conv_9',
                is_train=is_train)  # (7,7,1024)

            net = group_depthwise_conv2d(
                net, (3, 3), (1, 1),
                num_groups=1024,
                act=tf.nn.relu6,
                name='conv_10_dw',
                is_train=is_train)  # (7,7,1024)
            net = conv2d(
                net, (1, 1),
                1024, (1, 1),
                act=tf.nn.relu6,
                name='conv_10',
                is_train=is_train)  # (7,7,1024)

        with tf.variable_scope('gdc'):
            net = group_conv2d_nobias(
                net, (7, 7),
                512, (1, 1),
                num_groups=512,
                name='conv',
                is_train=is_train)  # (1,1,512)
            self.embedding = dense(
                flatten(net), 128, name='embedding', is_train=is_train)
