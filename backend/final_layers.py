from backend.layers import *


def gdc(net):
    input_node = net.arch_output
    is_train = net.is_train

    with tf.variable_scope('gdc'):
        net = conv2d(  # (1,1,512)
            input_node, (7, 7),
            512, (1, 1),
            name='conv',
            is_train=is_train)
        net = dense(flatten(net), 128, name='embedding', is_train=is_train)
    return net


def g_type(net):
    input_node = net.arch_output
    is_train = net.is_train

    with tf.variable_scope('g_type'):
        net = tf.layers.batch_normalization(input_node, name='bn', training=is_train)
        net = dense(flatten(net), 128, name='embedding', is_train=is_train)

    return net
