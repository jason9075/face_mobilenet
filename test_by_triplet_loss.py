import os

import cv2

import utils
import numpy as np
import tensorflow as tf

CKPT_NAME = 'RES_NET50_iter_4000.ckpt'
SHAPE = (112, 112)
INPUT_NODE = 'input_images'
EMB_NODE = 'valid/net/l2_embeddings'


def eval_by_ckpt(verification_name):
    verification_path = os.path.join('tfrecord', verification_name)
    ver_dataset = utils.get_ver_data(verification_path, SHAPE)

    with tf.Session() as sess:
        saver = tf.train.import_meta_graph(
            f'model_out/{CKPT_NAME}.meta', clear_devices=True)
        saver.restore(sess, f"model_out/{CKPT_NAME}")

        input_tensor = tf.get_default_graph().get_tensor_by_name(
            f"{INPUT_NODE}:0")
        embedding_tensor = tf.get_default_graph().get_tensor_by_name(
            f"{EMB_NODE}:0")

        img1 = cv2.imread('images/train_image/0_2924051/9.jpg')
        img1 = cv2.resize(img1, SHAPE)
        img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
        img1 = img1 - 127.5
        img1 *= 0.0078125
        v1 = sess.run(embedding_tensor, feed_dict={input_tensor: [img1]})[0]

        img2 = cv2.imread('images/train_image/0_2924051/0.jpg')
        img2 = cv2.resize(img2, SHAPE)
        img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)
        img2 = img2 - 127.5
        img2 *= 0.0078125
        v2 = sess.run(embedding_tensor, feed_dict={input_tensor: [img2]})[0]
        print(np.linalg.norm(v1 - v2))
        exit(0)

        val_acc, val_thr = utils.ver_test(
            data_set=ver_dataset,
            sess=sess,
            l2_embedding_tensor=embedding_tensor,
            input_placeholder=input_tensor)
        print('test accuracy is: {}, thr: {}'.format(val_acc, val_thr))


def eval_by_dlib(verification_name):
    verification_path = os.path.join('tfrecord', verification_name)
    ver_dataset = utils.get_ver_data(verification_path, SHAPE, preprocessing=False)

    val_acc, val_thr = utils.ver_dlib(data_set=ver_dataset)
    print('test accuracy is: {}, thr: {}'.format(val_acc, val_thr))


if __name__ == '__main__':
    eval_by_ckpt('verification.tfrecord')
    # eval_by_dlib('ver_marathon.tfrecord')
