from backend.arch.basenet import BaseNet
from backend.layers import *


class ResNet34(BaseNet):
    def __init__(self, input_layer, is_train):
        super().__init__(input_layer, is_train)  # input = n
        self.layer_list = [3, 4, 6, 3]
        with tf.variable_scope('resnet34'):
            ch = 32  # paper is 64. (total parameters: 64 have 21492736, 32 have 5498368)
            net = tf.layers.conv2d(input_layer, ch, 7, strides=2, name='conv')  # (n/2,n/2,64)
            net = max_pool(net, filter_size=(3, 3), strides=(2, 2), name='max_pool')  # (n/4,n/4,64)

            for i in range(self.layer_list[0]):
                net = resblock(net, ch=ch, is_train=is_train, down_sample=False, scope='resblock0_' + str(i))

            net = resblock(net, ch=ch * 2, is_train=is_train, down_sample=True,
                           scope='resblock1_0')  # (n/8,n/8,128)

            for i in range(1, self.layer_list[1]):
                net = resblock(net, ch=ch * 2, is_train=is_train, down_sample=False, scope='resblock1_' + str(i))

            net = resblock(net, ch=ch * 4, is_train=is_train, down_sample=True,
                           scope='resblock2_0')  # (n/16,n/16,256)

            for i in range(1, self.layer_list[2]):
                net = resblock(net, ch=ch * 4, is_train=is_train, down_sample=False, scope='resblock2_' + str(i))

            net = resblock(net, ch=ch * 8, is_train=is_train, down_sample=True,
                           scope='resblock_3_0')  # (n/32,n/32,512)

            for i in range(1, self.layer_list[3]):
                net = resblock(net, ch=ch * 8, is_train=is_train, down_sample=False, scope='resblock_3_' + str(i))

            net = tf.reduce_mean(net, axis=[1, 2], keepdims=True, name='global_average')

            self.arch_output = net
