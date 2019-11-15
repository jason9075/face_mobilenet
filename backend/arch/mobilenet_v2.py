from backend.arch.basenet import BaseNet
from backend.layers import *


class MobileNetV2(BaseNet):
    def __init__(self, input_layer, is_train):
        super().__init__(input_layer, is_train)
        exp = 6  # expansion ratio
        with tf.variable_scope('mobilenet_v2'):
            net = conv2d(  # (n/2,n/2,24)
                input_layer, (3, 3),
                32, (2, 2),
                act=ACT_FUNC,
                name='conv_1',
                is_train=is_train)

            net = inv_res_block(
                net,
                1,
                16, (1, 1),
                name='res2_1',
                is_train=is_train)

            net = inv_res_block(  # (n/4,n/4,24)
                net,
                exp,
                24, (2, 2),
                name='res3_1',
                is_train=is_train)
            net = inv_res_block(
                net, exp, 24, (1, 1), name='res3_2', is_train=is_train)

            net = inv_res_block(  # (n/8,n/8,32)
                net,
                exp,
                32, (2, 2),
                name='res4_1',
                is_train=is_train)
            net = inv_res_block(
                net, exp, 32, (1, 1), name='res4_2', is_train=is_train)
            net = inv_res_block(
                net, exp, 32, (1, 1), name='res4_3', is_train=is_train)

            net = inv_res_block(  # (n/16,n/16,64)
                net,
                exp,
                64, (2, 2),
                is_train=is_train,
                name='res5_1')
            net = inv_res_block(
                net, exp, 64, (1, 1), name='res5_2', is_train=is_train)
            net = inv_res_block(
                net, exp, 64, (1, 1), name='res5_3', is_train=is_train)
            net = inv_res_block(
                net, exp, 64, (1, 1), name='res5_4', is_train=is_train)

            net = inv_res_block(
                net,
                exp,
                96, (1, 1),
                name='res6_1',
                is_train=is_train)
            net = inv_res_block(
                net, exp, 96, (1, 1), name='res6_2', is_train=is_train)
            net = inv_res_block(
                net, exp, 96, (1, 1), name='res6_3', is_train=is_train)

            net = inv_res_block(  # (n/32,n/32,160)
                net,
                exp,
                160, (2, 2),
                name='res7_1',
                is_train=is_train)
            net = inv_res_block(
                net, exp, 160, (1, 1), name='res7_2', is_train=is_train)
            net = inv_res_block(
                net, exp, 160, (1, 1), name='res7_3', is_train=is_train)

            net = inv_res_block(  # (n/32,n/32,320)
                net,
                exp,
                320, (1, 1),
                name='res8_1',
                is_train=is_train,
                shortcut=False)

            net = conv2d(  # (n/32,n/32,1024)
                net, (1, 1),
                1024, (1, 1),
                act=ACT_FUNC,
                name='p_wise',
                is_train=is_train)

            self.arch_output = net
