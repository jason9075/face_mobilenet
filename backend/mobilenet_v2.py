from backend.layers import *


class MobileNetV2:
    def __init__(self, input_layer):
        with tf.variable_scope('mobilenet_v2'):
            net = conv2d(input_layer, (3, 3), 32, (2, 2), act=tf.nn.relu6, name='conv_1')  # (112,112,32)

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

        with tf.variable_scope('gdc'):
            net = conv_gdc(net, (7, 7), 512, (1, 1), name='conv_gdc')  # (1,1,512)
            self.embedding = dense(flatten(net), 128, name='embedding')
