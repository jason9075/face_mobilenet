from backend.arch.basenet import BaseNet
from backend.layers import *


class ResNet50(BaseNet):
    def __init__(self, input_layer, is_train):
        super().__init__(input_layer, is_train)  # input = n
        self.layer_list = [3, 4, 6, 3]
        with tf.variable_scope('resnet50'):
            ch = 64  # paper is 64. (total parameters: 64 have 38516608, 32 have 9826240)
            net = tf.layers.conv2d(input_layer, ch, 7, strides=2, name='res1')  # (n/2,n/2,64)
            net = max_pool(net, filter_size=(3, 3), strides=(2, 2), name='res2_max_pool')  # (n/4,n/4,64)

            for i in range(self.layer_list[0]):
                net = bottle_resblock(net, ch=ch, is_train=is_train, down_sample=False,
                                      scope='res2_' + str(i))

            net = bottle_resblock(net, ch=ch * 2, is_train=is_train, down_sample=True,
                                  scope='res3_0')  # (n/8,n/8,128)

            for i in range(1, self.layer_list[1]):
                net = bottle_resblock(net, ch=ch * 2, is_train=is_train, down_sample=False,
                                      scope='res3_' + str(i))

            net = bottle_resblock(net, ch=ch * 4, is_train=is_train, down_sample=True,
                                  scope='res4_0')  # (n/16,n/16,256)

            for i in range(1, self.layer_list[2]):
                net = bottle_resblock(net, ch=ch * 4, is_train=is_train, down_sample=False,
                                      scope='res4_' + str(i))

            net = bottle_resblock(net, ch=ch * 8, is_train=is_train, down_sample=True,
                                  scope='res5_0')  # (n/32,n/32,512)

            for i in range(1, self.layer_list[3]):
                net = bottle_resblock(net, ch=ch * 8, is_train=is_train, down_sample=False,
                                      scope='res5_' + str(i))

            net = tf.reduce_mean(net, axis=[1, 2], keepdims=True, name='global_average')

            self.arch_output = net
