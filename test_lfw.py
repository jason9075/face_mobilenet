import pickle

import cv2
import mxnet as mx
import numpy as np
import tensorflow as tf

import utils

INPUT_SIZE = (224, 224)
LFW_PATH = 'tfrecord/lfw.bin'
CKPT_NAME = 'RES_NET50_iter_462000.ckpt'
INPUT_NODE = 'input_images:0'
TRAINING_NODE = 'is_training:0'
OUTPUT_NODE = 'g_type/embedding/Identity:0'


def load_bin(bin_path):
    bins, issame_list = pickle.load(open(bin_path, 'rb'), encoding='bytes')
    data_list = []
    for _ in [0, 1]:
        data = np.empty((len(issame_list) * 2, INPUT_SIZE[0], INPUT_SIZE[1], 3))
        data_list.append(data)
    for i in range(len(issame_list) * 2):
        _bin = bins[i]
        img = mx.image.imdecode(_bin).asnumpy()
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        img = cv2.resize(img, INPUT_SIZE)
        for flip in [0, 1]:
            if flip == 1:
                img = np.fliplr(img)
            data_list[flip][i, ...] = img
        i += 1
        if i % 1000 == 0:
            print('loading bin', i)
    print(data_list[0].shape)
    return data_list, issame_list


def main():
    lfw_set = load_bin(LFW_PATH)

    with tf.Session() as sess:
        saver = tf.train.import_meta_graph(
            f'model_out/{CKPT_NAME}.meta', clear_devices=True)
        saver.restore(sess, CKPT_NAME)

        input_tensor = tf.get_default_graph().get_tensor_by_name(
            INPUT_NODE)
        trainable = tf.get_default_graph().get_tensor_by_name(TRAINING_NODE)
        embedding_tensor = tf.get_default_graph().get_tensor_by_name(
            OUTPUT_NODE)

        feed_dict_test = {trainable: False}

        val_acc, val_thr = utils.lfw_test(
            data_set=lfw_set,
            sess=sess,
            embedding_tensor=embedding_tensor,
            feed_dict=feed_dict_test,
            input_placeholder=input_tensor)

        print('lfw acc: %.2f, thr: %.2f' % (val_acc, val_thr))


if __name__ == '__main__':
    main()
