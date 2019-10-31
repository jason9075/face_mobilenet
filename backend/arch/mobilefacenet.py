from backend.arch.basenet import BaseNet
from backend.layers import *


# ref: https://github.com/sirius-ai/MobileFaceNet_TF/blob/master/nets/MobileFaceNet.py
class MobileFaceNet(BaseNet):
    def __init__(self, input_layer, is_train):
        super().__init__(input_layer, is_train)
        with tf.variable_scope('mobilefacenet'):
            net = conv2d(  # (n/2,n/2,64)
                input_layer, (3, 3),
                64, (2, 2),
                act=ACT_FUNC,
                name='conv_1',
                is_train=is_train)

            net = depthwise_conv2d(
                net, (3, 3), (1, 1), name='dconv_2', is_train=is_train)

            net = inv_res_block(  # (n/4,n/4,64)
                net,
                2,
                64, (2, 2),
                name='res3_1',
                is_train=is_train)
            net = inv_res_block(
                net, 2, 64, (1, 1), name='res3_2', is_train=is_train)
            net = inv_res_block(
                net, 2, 64, (1, 1), name='res3_3', is_train=is_train)
            net = inv_res_block(
                net, 2, 64, (1, 1), name='res3_4', is_train=is_train)
            net = inv_res_block(
                net, 2, 64, (1, 1), name='res3_5', is_train=is_train)

            net = inv_res_block(  # (n/8,n/8,128)
                net,
                4,
                128, (2, 2),
                name='res4',
                is_train=is_train)

            net = inv_res_block(  # (n/8,n/8,128)
                net,
                2,
                128, (1, 1),
                name='res5_1',
                is_train=is_train)
            net = inv_res_block(
                net, 2, 128, (1, 1), name='res5_2', is_train=is_train)
            net = inv_res_block(
                net, 2, 128, (1, 1), name='res5_3', is_train=is_train)
            net = inv_res_block(
                net, 2, 128, (1, 1), name='res5_4', is_train=is_train)
            net = inv_res_block(
                net, 2, 128, (1, 1), name='res5_5', is_train=is_train)
            net = inv_res_block(
                net, 2, 128, (1, 1), name='res5_6', is_train=is_train)

            net = inv_res_block(  # (n/16,n/16,128)
                net,
                4,
                128, (2, 2),
                name='res6',
                is_train=is_train)

            net = inv_res_block(  # (n/16,n/16,128)
                net,
                2,
                128, (1, 1),
                name='res7_1',
                is_train=is_train)
            net = inv_res_block(
                net, 2, 128, (1, 1), name='res7_2', is_train=is_train)

            net = conv2d(  # (n/16,n/16,512)
                net, (1, 1),
                512, (1, 1),
                act=ACT_FUNC,
                name='p_wise',
                is_train=is_train)

            self.arch_output = net
