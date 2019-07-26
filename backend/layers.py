import platform

import numpy as np
import tensorflow as tf

if platform.system() == 'Linux':
    DEVICE = '/gpu:0'
else:
    DEVICE = '/cpu:0'
D_TYPE = tf.float32
CUDNN_ON_GPU = True
BIAS_INIT = tf.constant_initializer(0.0)
ONE_INIT = tf.constant_initializer(1.0)
WEIGHT_INIT = tf.truncated_normal_initializer(stddev=0.02)
REGULARIZER = tf.contrib.layers.l2_regularizer(5e-6)


def conv2d(x,
           kernel,
           num_filter,
           stride,
           bn=True,
           act=tf.identity,
           name='conv',
           padding='SAME',
           is_train=True):
    stride = [1, stride[0], stride[1], 1]
    pre_channel = int(x.get_shape()[-1])
    shape = [kernel[0], kernel[1], pre_channel, num_filter]
    with tf.device(DEVICE):
        with tf.variable_scope(name):
            w = tf.get_variable(
                name='w_conv',
                shape=shape,
                initializer=WEIGHT_INIT,
                dtype=D_TYPE,
                regularizer=REGULARIZER)
            out = tf.nn.conv2d(
                x, w, stride, padding, use_cudnn_on_gpu=CUDNN_ON_GPU)
            if bn:
                out = tf.layers.batch_normalization(
                    out, name='bn', training=is_train)
            return act(out)


def depthwise_conv2d(x,
                     kernel,
                     stride,
                     bn=True,
                     act=tf.identity,
                     name='conv_dw',
                     is_train=True):
    stride = [1, stride[0], stride[1], 1]
    num_filter = x.shape[-1]
    kernel_shape = [kernel[0], kernel[1], num_filter, 1]

    with tf.device(DEVICE):
        with tf.variable_scope(name):
            w = tf.get_variable(
                name='w_dw',
                shape=kernel_shape,
                initializer=WEIGHT_INIT,
                dtype=D_TYPE,
                regularizer=REGULARIZER)
            out = tf.nn.depthwise_conv2d(
                x, w, stride, name='d_conv', padding='SAME')
            if bn:
                out = tf.layers.batch_normalization(
                    out, name='bn', training=is_train)
            return act(out)


def group_conv2d(x,
                 kernel,
                 num_filter,
                 stride,
                 num_groups=1,
                 bn=True,
                 act=tf.identity,
                 name='group_conv',
                 padding='SAME',
                 is_train=True):
    pre_channel = int(x.get_shape()[-1])
    shape = [kernel[0], kernel[1], pre_channel // num_groups, num_filter]

    output_list = []

    with tf.device(DEVICE):
        with tf.variable_scope(name):
            w = tf.get_variable(
                name='w_conv',
                shape=shape,
                initializer=WEIGHT_INIT,
                dtype=D_TYPE,
                regularizer=REGULARIZER)
            input_list = tf.split(x, num_groups, axis=-1)
            filter_list = tf.split(w, num_groups, axis=-1)

            for conv_idx, (input_tensor, filter_tensor) in enumerate(
                    zip(input_list, filter_list)):
                conv = tf.nn.convolution(
                    input_tensor, filter_tensor, padding, stride, name=name)
                output_list.append(conv)
            out = tf.concat(output_list, axis=-1)
            if bn:
                out = tf.layers.batch_normalization(
                    out, name='bn', training=is_train)
            return act(out)


def flatten(x, name='flatten'):
    all_dims_exc_first = np.prod([v.value for v in x.get_shape()[1:]])
    out = tf.reshape(x, [-1, all_dims_exc_first], name=name)
    return out


def dense(x,
              num_classes,
              bn=True,
              act=tf.identity,
              name='dense',
              is_train=True):
    n_in = int(x.get_shape()[-1])
    with tf.device(DEVICE):
        with tf.variable_scope(name):
            w = tf.get_variable(
                name='w_dense',
                shape=[n_in, num_classes],
                initializer=WEIGHT_INIT,
                dtype=D_TYPE,
                regularizer=REGULARIZER)
            b = tf.get_variable(
                name='b_dense',
                shape=num_classes,
                initializer=BIAS_INIT,
                dtype=D_TYPE)
            out = tf.nn.bias_add(tf.matmul(x, w), b)
            if bn:
                out = tf.layers.batch_normalization(out, training=is_train, gamma_initializer=ONE_INIT)
            return act(out)


def inv_res_block(x, expansion_ratio, output_dim, stride, name='inv_res_block', is_train=True, shortcut=True):
    in_dim = int(x.get_shape()[-1])
    with tf.variable_scope(name):
        # pw
        bottleneck_dim = round(expansion_ratio * in_dim)
        net = conv2d(x, (1, 1), bottleneck_dim, stride, act=tf.nn.relu6, name='pw')
        # dw
        net = depthwise_conv2d(net, (3, 3), stride, act=tf.nn.relu6, name='dw', is_train=is_train)
        # pw & linear
        net = conv2d(net, (1, 1), output_dim, stride, act=tf.identity, name='pw_linear')

        # element wise add, only for stride==1
        if shortcut and stride == (1, 1):
            if in_dim != output_dim:
                ins = conv2d(x, (1, 1), output_dim, (1, 1), bn=False, act=tf.identity, name='shortcut')
                net = net + ins
            else:
                net = net + x

        return net


def global_avg(x):
    with tf.name_scope('global_avg'):
        net = tf.layers.average_pooling2d(x, x.get_shape()[1:-1], 1)
        return net
