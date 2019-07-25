from backend.layers import *


# ref: https://github.com/sirius-ai/MobileFaceNet_TF/blob/master/nets/MobileFaceNet.py
class MobileFaceNet:
    def __init__(self, input_layer, is_train):
        blocks = [1, 4, 6, 2]
        with tf.variable_scope('mobilefacenet'):
            net = conv2d(  # (56,56,64)
                input_layer, (3, 3),
                64, (2, 2),
                act=tf.nn.relu6,
                name='conv_1',
                is_train=is_train)

            net = depthwise_conv2d(net, (3, 3), (1, 1), name='dvpnv_2')

            net = inv_res_block(  # (28,28,64)
                net, 2, 64, (2, 2), is_train=is_train,
                name='res3_1')
            net = inv_res_block(
                net, 2, 64, (1, 1), is_train=is_train,
                name='res3_2')
            net = inv_res_block(
                net, 2, 64, (1, 1), is_train=is_train,
                name='res3_3')
            net = inv_res_block(
                net, 2, 64, (1, 1), is_train=is_train,
                name='res3_4')
            net = inv_res_block(
                net, 2, 64, (1, 1), is_train=is_train,
                name='res3_5')

            net = inv_res_block(  # (14,14,128)
                net, 4, 128, (2, 2), is_train=is_train,
                name='res4')

            net = inv_res_block(  # (14,14,128)
                net, 2, 128, (1, 1), is_train=is_train,
                name='res5_1')
            net = inv_res_block(
                net, 2, 128, (1, 1), is_train=is_train,
                name='res5_2')
            net = inv_res_block(
                net, 2, 128, (1, 1), is_train=is_train,
                name='res5_3')
            net = inv_res_block(
                net, 2, 128, (1, 1), is_train=is_train,
                name='res5_4')
            net = inv_res_block(
                net, 2, 128, (1, 1), is_train=is_train,
                name='res5_5')
            net = inv_res_block(
                net, 2, 128, (1, 1), is_train=is_train,
                name='res5_6')

            net = inv_res_block(  # (7,7,128)
                net, 4, 128, (2, 2), is_train=is_train,
                name='res6')

            net = inv_res_block(  # (7,7,128)
                net, 2, 128, (1, 1), is_train=is_train,
                name='res7_1')
            net = inv_res_block(
                net, 2, 128, (1, 1), is_train=is_train,
                name='res7_2')

            net = conv2d(  # (7,7,512)
                net, (1, 1),
                512, (1, 1),
                act=tf.nn.relu6,
                is_train=is_train,
                name='p_wise')

        with tf.variable_scope('gdc'):
            net = group_conv2d(  # (1,1,512)
                net, (7, 7),
                512, (1, 1),
                num_groups=512,
                name='conv',
                is_train=is_train)
            self.embedding = gdc_dense(
                flatten(net), 128, name='embedding', is_train=is_train)
