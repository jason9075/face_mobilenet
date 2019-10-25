from backend.arch.basenet import BaseNet
from backend.layers import *


class ResNet50(BaseNet):
    def __init__(self, input_layer, is_train):
        super().__init__(input_layer, is_train)  # input = n
        self.layer_list = [3, 4, 6, 3]
        with tf.variable_scope('resnet50'):
            ch = 32  # paper is 64
            net = tf.layers.conv2d(input_layer, ch, 3, strides=1, name='conv')  # (n,n,32)

            for i in range(self.layer_list[0]):
                net = resblock(net, channels=ch, is_train=is_train, down_sample=False, scope='resblock0_' + str(i))

            net = resblock(net, channels=ch * 2, is_train=is_train, down_sample=True,
                           scope='resblock1_0')  # (n/2,n/2,64)

            for i in range(1, self.layer_list[1]):
                net = resblock(net, channels=ch * 2, is_train=is_train, down_sample=False, scope='resblock1_' + str(i))

            net = resblock(net, channels=ch * 4, is_train=is_train, down_sample=True,
                           scope='resblock2_0')  # (n/4,n/4,128)

            for i in range(1, self.layer_list[2]):
                net = resblock(net, channels=ch * 4, is_train=is_train, down_sample=False, scope='resblock2_' + str(i))

            net = resblock(net, channels=ch * 8, is_train=is_train, down_sample=True,
                           scope='resblock_3_0')  # (n/8,n/8,256)

            for i in range(1, self.layer_list[3]):
                net = resblock(net, channels=ch * 8, is_train=is_train, down_sample=False, scope='resblock_3_' + str(i))

            net = tf.layers.batch_normalization(net, training=is_train, name='batch_norm')
            net = tf.nn.relu(net)

            net = tf.reduce_mean(net, axis=[1, 2], keepdims=True, name='global_average')

            self.arch_output = net
