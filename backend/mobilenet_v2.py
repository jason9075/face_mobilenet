from backend.layers import *


class MobileNetV2:
    def __init__(self, input_layer, is_train):
        exp = 6  # expansion ratio
        with tf.variable_scope('mobilenet_v2'):
            net = conv2d(
                input_layer, (3, 3),
                32, (1, 1),
                act=tf.nn.relu6,
                name='conv_1',
                is_train=is_train)  # (112,112,32)

            net = res_block(net, 1, 16, (1, 1), is_train=is_train, name='res2_1')  # (112,112,16)

            net = res_block(net, exp, 24, (2, 2), is_train=is_train, name='res3_1')  # (56,56,24)
            net = res_block(net, exp, 24, (1, 1), is_train=is_train, name='res3_2')

            net = res_block(net, exp, 32, (2, 2), is_train=is_train, name='res4_1')  # (28,28,32)
            net = res_block(net, exp, 32, (1, 1), is_train=is_train, name='res4_2')
            net = res_block(net, exp, 32, (1, 1), is_train=is_train, name='res4_3')

            net = res_block(net, exp, 64, (2, 2), is_train=is_train, name='res5_1')  # (14,14,64)
            net = res_block(net, exp, 64, (1, 1), is_train=is_train, name='res5_2')
            net = res_block(net, exp, 64, (1, 1), is_train=is_train, name='res5_3')
            net = res_block(net, exp, 64, (1, 1), is_train=is_train, name='res5_4')

            net = res_block(net, exp, 96, (1, 1), is_train=is_train, name='res6_1')  # (14,14,96)
            net = res_block(net, exp, 96, (1, 1), is_train=is_train, name='res6_2')
            net = res_block(net, exp, 96, (1, 1), is_train=is_train, name='res6_3')

            net = res_block(net, exp, 160, (2, 2), is_train=is_train, name='res7_1')  # (7,7,160)
            net = res_block(net, exp, 160, (1, 1), is_train=is_train, name='res7_2')
            net = res_block(net, exp, 160, (1, 1), is_train=is_train, name='res7_3')

            net = res_block(net, exp, 320, (1, 1), is_train=is_train, name='res8_1', shortcut=False)  # (7,7,320)

            net = conv2d(net, (1, 1), 1024, (1, 1), act=tf.nn.relu6, is_train=is_train, name='p_wise')  # (7,7,1024)

        with tf.variable_scope('gdc'):
            net = group_conv2d(
                net, (7, 7),
                512, (1, 1),
                num_groups=512,
                name='conv',
                is_train=is_train)  # (1,1,512)
            self.embedding = gdc_dense(
                flatten(net), 128, name='embedding', is_train=is_train)
