import os

import utils
import numpy as np
import tensorflow as tf
import face_recognition as fr

CKPT_NAME = 'InsightFace_iter_85000.ckpt'
SHAPE = (224, 224)


def eval_by_ckpt(verification_name):
    verification_path = os.path.join('tfrecord', verification_name)
    ver_dataset = utils.get_ver_data(verification_path, SHAPE)

    with tf.Session() as sess:
        saver = tf.train.import_meta_graph(
            f'model_out/{CKPT_NAME}.meta', clear_devices=True)
        saver.restore(sess, f"model_out/{CKPT_NAME}")

        input_tensor = tf.get_default_graph().get_tensor_by_name(
            "input_images:0")
        embedding_tensor = tf.get_default_graph().get_tensor_by_name(
            "gdc/embedding/Identity:0")

        feed_dict_test = {}
        val_acc, val_thr = utils.ver_test(
            data_set=ver_dataset,
            sess=sess,
            embedding_tensor=embedding_tensor,
            feed_dict=feed_dict_test,
            input_placeholder=input_tensor)
        print('test accuracy is: {}, thr: {}'.format(val_acc, val_thr))


def eval_by_dlib(verification_name):
    verification_path = os.path.join('tfrecord', verification_name)
    ver_dataset = utils.get_ver_data(verification_path, SHAPE, preprocessing=False)

    val_acc, val_thr = ver_dlib(data_set=ver_dataset)
    print('test accuracy is: {}, thr: {}'.format(val_acc, val_thr))


def ver_dlib(data_set):
    first_list, second_list, true_same = data_set[0], data_set[1], np.array(data_set[2])

    dist_list = []
    for first, second in zip(first_list, second_list):
        h, w, _ = first.shape
        first_vector = fr.face_encodings(first, [(0, w, h, 0)])[0]
        second_vector = fr.face_encodings(second, [(0, w, h, 0)])[0]

        vector_pair = utils.preprocessing.normalize([first_vector, second_vector])
        dist = np.linalg.norm(vector_pair[0] - vector_pair[1])
        dist_list.append(dist)

    thresholds = np.arange(0.1, 4.0, 0.1)

    accs = []
    for threshold in thresholds:
        pred_same = np.less(dist_list, threshold)
        tp = np.sum(np.logical_and(pred_same, true_same))
        tn = np.sum(np.logical_and(np.logical_not(pred_same), np.logical_not(true_same)))
        acc = float(tp + tn) / len(first_list)
        accs.append(acc)
    best_threshold_index = int(np.argmax(accs))

    return accs[best_threshold_index], thresholds[best_threshold_index]


if __name__ == '__main__':
    eval_by_ckpt('ver_marathon.tfrecord')
    # eval_by_dlib('ver_marathon.tfrecord')
