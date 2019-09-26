from backend.arch.basenet import BaseNet
from backend.layers import *


class MobileNetV2(BaseNet):
    def __init__(self, input_layer, is_train):
        super().__init__(input_layer, is_train)
        exp = 6  # expansion ratio
        with tf.variable_scope('mobilenet_v2'):
            net = conv2d(  # (112,112,32)
                input_layer, (3, 3),
                32, (1, 1),
                act=ACT_FUNC,
                name='conv_1',
                is_train=is_train)

            net = inv_res_block(  # (112,112,16)
                net,
                1,
                16, (1, 1),
                name='res2_1',
                is_train=is_train)

            net = inv_res_block(  # (56,56,24)
                net,
                exp,
                24, (2, 2),
                name='res3_1',
                is_train=is_train)
            net = inv_res_block(
                net, exp, 24, (1, 1), name='res3_2', is_train=is_train)

            net = inv_res_block(  # (28,28,32)
                net,
                exp,
                32, (2, 2),
                name='res4_1',
                is_train=is_train)
            net = inv_res_block(
                net, exp, 32, (1, 1), name='res4_2', is_train=is_train)
            net = inv_res_block(
                net, exp, 32, (1, 1), name='res4_3', is_train=is_train)

            net = inv_res_block(  # (14,14,64)
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

            net = inv_res_block(  # (14,14,96)
                net,
                exp,
                96, (1, 1),
                name='res6_1',
                is_train=is_train)
            net = inv_res_block(
                net, exp, 96, (1, 1), name='res6_2', is_train=is_train)
            net = inv_res_block(
                net, exp, 96, (1, 1), name='res6_3', is_train=is_train)

            net = inv_res_block(  # (7,7,160)
                net,
                exp,
                160, (2, 2),
                name='res7_1',
                is_train=is_train)
            net = inv_res_block(
                net, exp, 160, (1, 1), name='res7_2', is_train=is_train)
            net = inv_res_block(
                net, exp, 160, (1, 1), name='res7_3', is_train=is_train)

            net = inv_res_block(  # (7,7,320)
                net,
                exp,
                320, (1, 1),
                name='res8_1',
                is_train=is_train,
                shortcut=False)

            net = conv2d(  # (7,7,1024)
                net, (1, 1),
                1024, (1, 1),
                act=ACT_FUNC,
                name='p_wise',
                is_train=is_train)

            self.arch_output = net
