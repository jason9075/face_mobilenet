from backend.layers import *


class MobileNetV1:
    def __init__(self, input_layer, is_train):
        bf = 32
        with tf.variable_scope('mobilenet_v2'):
            net = conv2d(
                input_layer, (3, 3),
                bf, (1, 1),
                act=tf.nn.relu6,
                name='conv_1',
                is_train=is_train)  # (112,112,32)
            # PS default stride is (2,2) but our training dataset (w,h) is 112, so stride is 1 here.

            net = group_conv2d(
                net, (3, 3), bf * 2, (1, 1),
                num_groups=bf,
                act=tf.nn.relu6,
                name='conv_2_dw',
                is_train=is_train)  # (112,112,32)
            net = conv2d(
                net, (1, 1),
                bf * 2, (1, 1),
                act=tf.nn.relu6,
                name='conv_2',
                is_train=is_train)  # (112,112,32)

            net = group_conv2d(
                net, (3, 3), bf * 2, (2, 2),
                num_groups=bf * 2,
                act=tf.nn.relu6,
                name='conv_3_dw',
                is_train=is_train)  # (112,112,64)
            net = conv2d(
                net, (1, 1),
                bf * 4, (1, 1),
                act=tf.nn.relu6,
                name='conv_3',
                is_train=is_train)  # (56,56,64)

            net = group_conv2d(
                net, (3, 3), bf * 4, (1, 1),
                num_groups=bf * 4,
                act=tf.nn.relu6,
                name='conv_4_dw',
                is_train=is_train)  # (56,56,128)
            net = conv2d(
                net, (1, 1),
                bf * 4, (1, 1),
                act=tf.nn.relu6,
                name='conv_4',
                is_train=is_train)  # (56,56,128)

            net = group_conv2d(
                net, (3, 3), bf * 4, (2, 2),
                num_groups=bf * 4,
                act=tf.nn.relu6,
                name='conv_5_dw',
                is_train=is_train)  # (56,56,128)
            net = conv2d(
                net, (1, 1),
                bf * 8, (1, 1),
                act=tf.nn.relu6,
                name='conv_5',
                is_train=is_train)  # (28,28,128)

            net = group_conv2d(
                net, (3, 3), bf * 8, (1, 1),
                num_groups=bf * 8,
                act=tf.nn.relu6,
                name='conv_6_dw',
                is_train=is_train)  # (28,28,256)
            net = conv2d(
                net, (1, 1),
                bf * 8, (1, 1),
                act=tf.nn.relu6,
                name='conv_6',
                is_train=is_train)  # (28,28,256)

            net = group_conv2d(
                net, (3, 3), bf * 8, (2, 2),
                num_groups=bf * 8,
                act=tf.nn.relu6,
                name='conv_7_dw',
                is_train=is_train)  # (28,28,256)
            net = conv2d(
                net, (1, 1),
                bf * 16, (1, 1),
                act=tf.nn.relu6,
                name='conv_7',
                is_train=is_train)  # (14,14,256)

            net = group_conv2d(
                net, (3, 3), bf * 16, (1, 1),
                num_groups=bf * 16,
                act=tf.nn.relu6,
                name='conv_8_1_dw',
                is_train=is_train)  # (14,14,512)
            net = conv2d(
                net, (1, 1),
                bf * 16, (1, 1),
                act=tf.nn.relu6,
                name='conv_8_1',
                is_train=is_train)
            net = group_conv2d(
                net, (3, 3), bf * 16, (1, 1),
                num_groups=bf * 16,
                act=tf.nn.relu6,
                name='conv_8_2_dw',
                is_train=is_train)
            net = conv2d(
                net, (1, 1),
                bf * 16, (1, 1),
                act=tf.nn.relu6,
                name='conv_8_2',
                is_train=is_train)
            net = group_conv2d(
                net, (3, 3), bf * 16, (1, 1),
                num_groups=bf * 16,
                act=tf.nn.relu6,
                name='conv_8_3_dw',
                is_train=is_train)
            net = conv2d(
                net, (1, 1),
                bf * 16, (1, 1),
                act=tf.nn.relu6,
                name='conv_8_3',
                is_train=is_train)
            net = group_conv2d(
                net, (3, 3), bf * 16, (1, 1),
                num_groups=bf * 16,
                act=tf.nn.relu6,
                name='conv_8_4_dw',
                is_train=is_train)
            net = conv2d(
                net, (1, 1),
                bf * 16, (1, 1),
                act=tf.nn.relu6,
                name='conv_8_4',
                is_train=is_train)
            net = group_conv2d(
                net, (3, 3), bf * 16, (1, 1),
                num_groups=bf * 16,
                act=tf.nn.relu6,
                name='conv_8_5_dw',
                is_train=is_train)
            net = conv2d(
                net, (1, 1),
                bf * 16, (1, 1),
                act=tf.nn.relu6,
                name='conv_8_5',
                is_train=is_train)

            net = group_conv2d(
                net, (3, 3), bf * 16, (2, 2),
                num_groups=bf * 16,
                act=tf.nn.relu6,
                name='conv_9_dw',
                is_train=is_train)  # (14,14,512)
            net = conv2d(
                net, (1, 1),
                bf * 32, (1, 1),
                act=tf.nn.relu6,
                name='conv_9',
                is_train=is_train)  # (7,7,1024)

            net = group_conv2d(
                net, (3, 3), bf * 32, (1, 1),
                num_groups=bf * 32,
                act=tf.nn.relu6,
                name='conv_10_dw',
                is_train=is_train)  # (7,7,1024)
            net = conv2d(
                net, (1, 1),
                bf * 32, (1, 1),
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
