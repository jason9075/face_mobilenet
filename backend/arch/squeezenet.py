from backend.arch.basenet import BaseNet
from backend.layers import *


class SqueezeNet(BaseNet):
    def __init__(self, input_layer, is_train):
        super().__init__(input_layer, is_train)
        with tf.variable_scope('squeeze_net'):
            net = conv2d_bias(  # (112,112,96)
                input_layer, (7, 7),
                96, (1, 1),
                bn=False,
                name='conv_1')

            net = max_pool(  # (56,56,96)
                net,
                filter_size=(3, 3),
                strides=(2, 2),
                name='maxpool_1')

            net = fire_module(net, s1=16, e1=64, e3=64, name='fire2')  # (56,56,128)
            net = fire_module(net, s1=16, e1=64, e3=64, name='fire3')  # (56,56,128)
            net = fire_module(net, s1=32, e1=128, e3=128, name='fire4')  # (56,56,256)

            net = max_pool(  # (28,28,256)
                net,
                filter_size=(3, 3),
                strides=(2, 2),
                name='maxpool_4')

            net = fire_module(net, s1=32, e1=128, e3=128, name='fire5')  # (28,28,256)
            net = fire_module(net, s1=48, e1=192, e3=192, name='fire6')  # (28,28,384)
            net = fire_module(net, s1=48, e1=192, e3=192, name='fire7')  # (28,28,384)
            net = fire_module(net, s1=64, e1=256, e3=256, name='fire8')  # (28,28,512)

            net = max_pool(  # (14,14,512)
                net,
                filter_size=(3, 3),
                strides=(2, 2),
                name='maxpool_8')

            net = fire_module(net, s1=64, e1=256, e3=256, name='fire9')  # (14,14,512)

            net = tf.layers.dropout(net, training=is_train, name='dropout')

            net = conv2d_bias(  # (14,14,512)
                net, (1, 1),
                512, (1, 1),
                bn=False,
                name='conv_10')

            net = avg_pool(net, (14, 14), (2, 2), name='avg_pool10')  # (7,7,512)

            self.arch_output = net
