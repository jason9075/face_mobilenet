import numpy as np
import tensorflow as tf

DEVICE = '/cpu:0'
D_TYPE = tf.float32
CUDNN_ON_GPU = True
BIAS_INIT = tf.constant_initializer(0.0)
WEIGHT_INIT = tf.truncated_normal_initializer(stddev=0.02)


class MobileNetV2:
    def __init__(self, num_classes, batch_size=32):
        self.input_layer = tf.placeholder(D_TYPE, [batch_size, 224, 224, 3])  # Input Size: (224,224,3)

        with tf.variable_scope('mobilenet_v2'):
            net = conv2d(self.input_layer, (3, 3), 32, (2, 2), act=tf.nn.relu6, name='conv_1')  # (112,112,32)

            net = depthwise_conv2d(net, (3, 3), (1, 1), act=tf.nn.relu6, name='conv_2_dw')  # (112,112,32)
            net = conv2d(net, (1, 1), 64, (1, 1), act=tf.nn.relu6, name='conv_2')  # (112,112,32)

            net = depthwise_conv2d(net, (3, 3), (2, 2), act=tf.nn.relu6, name='conv_3_dw')  # (112,112,64)
            net = conv2d(net, (1, 1), 128, (1, 1), act=tf.nn.relu6, name='conv_3')  # (56,56,64)

            net = depthwise_conv2d(net, (3, 3), (1, 1), act=tf.nn.relu6, name='conv_4_dw')  # (56,56,128)
            net = conv2d(net, (1, 1), 128, (1, 1), act=tf.nn.relu6, name='conv_4')  # (56,56,128)

            net = depthwise_conv2d(net, (3, 3), (2, 2), act=tf.nn.relu6, name='conv_5_dw')  # (56,56,128)
            net = conv2d(net, (1, 1), 256, (1, 1), act=tf.nn.relu6, name='conv_5')  # (28,28,128)

            net = depthwise_conv2d(net, (3, 3), (1, 1), act=tf.nn.relu6, name='conv_6_dw')  # (28,28,256)
            net = conv2d(net, (1, 1), 256, (1, 1), act=tf.nn.relu6, name='conv_6')  # (28,28,256)

            net = depthwise_conv2d(net, (3, 3), (2, 2), act=tf.nn.relu6, name='conv_7_dw')  # (28,28,256)
            net = conv2d(net, (1, 1), 512, (1, 1), act=tf.nn.relu6, name='conv_7')  # (14,14,256)

            net = depthwise_conv2d(net, (3, 3), (1, 1), act=tf.nn.relu6, name='conv_8_1_dw')  # (14,14,512)
            net = conv2d(net, (1, 1), 512, (1, 1), act=tf.nn.relu6, name='conv_8_1')
            net = depthwise_conv2d(net, (3, 3), (1, 1), act=tf.nn.relu6, name='conv_8_2_dw')
            net = conv2d(net, (1, 1), 512, (1, 1), act=tf.nn.relu6, name='conv_8_2')
            net = depthwise_conv2d(net, (3, 3), (1, 1), act=tf.nn.relu6, name='conv_8_3_dw')
            net = conv2d(net, (1, 1), 512, (1, 1), act=tf.nn.relu6, name='conv_8_3')
            net = depthwise_conv2d(net, (3, 3), (1, 1), act=tf.nn.relu6, name='conv_8_4_dw')
            net = conv2d(net, (1, 1), 512, (1, 1), act=tf.nn.relu6, name='conv_8_4')
            net = depthwise_conv2d(net, (3, 3), (1, 1), act=tf.nn.relu6, name='conv_8_5_dw')
            net = conv2d(net, (1, 1), 512, (1, 1), act=tf.nn.relu6, name='conv_8_5')

            net = depthwise_conv2d(net, (3, 3), (2, 2), act=tf.nn.relu6, name='conv_9_dw')  # (14,14,512)
            net = conv2d(net, (1, 1), 1024, (1, 1), act=tf.nn.relu6, name='conv_9')  # (7,7,512)

            net = depthwise_conv2d(net, (3, 3), (1, 1), act=tf.nn.relu6, name='conv_10_dw')  # (7,7,1024)
            net = conv2d(net, (1, 1), 1024, (1, 1), act=tf.nn.relu6, name='conv_10')  # (7,7,1024)

        self.embedding = fc(net, (7, 7), 128, (1, 1), name='embedding')  # (1,1,128)

        self.result = dense(flatten(self.embedding), num_classes, name='dense_result')

        tf.summary.FileWriter("./", graph=tf.get_default_graph())


def conv2d(x, kernel, num_filter, stride, bn=True, act=tf.identity, name='conv', padding='SAME', is_train=True):
    stride = [1, stride[0], stride[1], 1]
    pre_channel = int(x.get_shape()[-1])
    shape = [kernel[0], kernel[1], pre_channel, num_filter]
    with tf.device(DEVICE):
        with tf.variable_scope(name):
            w = tf.get_variable(name='w_conv', shape=shape, initializer=WEIGHT_INIT, dtype=D_TYPE)
            b = tf.get_variable(name='b_conv', shape=(shape[-1]), initializer=BIAS_INIT, dtype=D_TYPE)
            conv = tf.nn.conv2d(x, w, stride, padding, use_cudnn_on_gpu=CUDNN_ON_GPU)
            out = tf.nn.bias_add(conv, b)
            if bn:
                out = tf.layers.batch_normalization(out, name='bn', training=is_train)
            return act(out)


def depthwise_conv2d(x, kernel, stride, bn=True, act=tf.identity, name='conv_dw', is_train=True):
    stride = [1, stride[0], stride[1], 1]
    num_filter = x.shape[-1]
    kernel_shape = [kernel[0], kernel[1], num_filter, 1]

    with tf.device(DEVICE):
        with tf.variable_scope(name):
            w = tf.get_variable(name='w_dw', shape=kernel_shape, initializer=WEIGHT_INIT, dtype=D_TYPE)
            b = tf.get_variable(name='b_dw', shape=(x.shape[-1]), initializer=BIAS_INIT, dtype=D_TYPE)
            conv = tf.nn.depthwise_conv2d(x, w, stride, name='d_conv', padding='SAME')
            out = tf.nn.bias_add(conv, b)
            if bn:
                out = tf.layers.batch_normalization(out, name='bn', training=is_train)
            return act(out)


def fc(x, kernel, num_filter, stride, name='fc', padding='SAME', is_train=True):
    stride = [1, stride[0], stride[1], 1]
    pre_channel = int(x.get_shape()[-1])
    shape = [kernel[0], kernel[1], pre_channel, num_filter]
    with tf.device(DEVICE):
        with tf.variable_scope(name):
            w = tf.get_variable(name='w_conv', shape=shape, initializer=WEIGHT_INIT, dtype=D_TYPE)
            conv = tf.nn.conv2d(x, w, stride, padding, use_cudnn_on_gpu=CUDNN_ON_GPU)
            out = tf.layers.batch_normalization(conv, name='vector', training=is_train)
            return out


def flatten(x):
    all_dims_exc_first = np.prod([v.value for v in x.get_shape()[1:]])
    out = tf.reshape(x, [-1, all_dims_exc_first])
    return out


def dense(x, num_classes, bn=True, act=tf.identity, name='dense', is_train=True):
    n_in = int(x.get_shape()[-1])
    with tf.device(DEVICE):
        with tf.variable_scope(name):
            w = tf.get_variable(name='w_dense', shape=[n_in, num_classes], initializer=WEIGHT_INIT, dtype=D_TYPE)
            b = tf.get_variable(name='b_dense', shape=num_classes, initializer=BIAS_INIT, dtype=D_TYPE)
            out = tf.nn.bias_add(tf.matmul(x, w), b)
            if bn:
                out = tf.layers.batch_normalization(out, training=is_train)
            return act(out)
