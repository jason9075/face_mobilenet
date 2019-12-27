import datetime
import os
import pickle
import timeit

import cv2
import numpy as np
import sklearn
import tensorflow as tf
from sklearn import preprocessing


def parse_function(example_proto):
    features = {'image_raw': tf.io.FixedLenFeature([], tf.string),
                'label': tf.io.FixedLenFeature([], tf.int64)}
    features = tf.io.parse_single_example(example_proto, features)
    img = tf.image.decode_jpeg(features['image_raw'], dct_method='INTEGER_ACCURATE')
    img = tf.image.random_brightness(img, 0.2)
    img = tf.image.random_saturation(img, 0.6, 1.6)
    img = tf.image.random_contrast(img, 0.6, 1.4)
    img = tf.image.random_flip_left_right(img)
    img = tf_pre_process_image(img, (224, 224))
    label = tf.cast(features['label'], tf.int64)
    return img, label


def tf_pre_process_image(img, shape):
    img = tf.reshape(img, shape=(shape[0], shape[1], 3))
    # r, g, b = tf.split(img, num_or_size_splits=3, axis=-1)
    # img = tf.concat([b, g, r], axis=-1)
    img = tf.cast(img, dtype=tf.float32)
    img = tf.subtract(img, 127.5)
    img = tf.multiply(img, 0.0078125)
    return img


def pre_process_image(img, shape):
    img = cv2.resize(img, shape)
    img = np.array(img, dtype=np.float32)
    img -= 127.5
    img *= 0.0078125
    return img


def get_ver_data(record_path, shape, preprocessing=True):
    record_iterator = tf.python_io.tf_record_iterator(path=record_path)
    first_list = []
    second_list = []
    is_same_list = []
    for record in record_iterator:
        example = tf.train.Example()
        example.ParseFromString(record)
        image_string = example.features.feature['image_first'].bytes_list.value[0]
        img = np.fromstring(image_string, dtype=np.uint8)
        img_first = cv2.imdecode(img, cv2.IMREAD_COLOR)
        if preprocessing:
            img_first = pre_process_image(img_first, shape)
        first_list.append(img_first)

        image_string = example.features.feature['image_second'].bytes_list.value[0]
        img = np.fromstring(image_string, dtype=np.uint8)
        img_second = cv2.imdecode(img, cv2.IMREAD_COLOR)
        if preprocessing:
            img_second = pre_process_image(img_second, shape)
        second_list.append(img_second)

        is_same = example.features.feature['is_same'].int64_list.value[0]
        is_same_list.append(is_same)

    return [first_list, second_list, is_same_list]


def ver_tfrecord(data_set, embedding_fn):
    first_list, second_list, true_same = data_set[0], data_set[1], np.array(data_set[2])
    total = len(true_same)
    same = int(np.sum(true_same))
    diff = total - same
    print('samples: %d, same: %d, diff: %d' % (total, same, diff))

    dist_list = []
    start = timeit.default_timer()
    for idx, (first, second) in enumerate(zip(first_list, second_list)):
        result1, result2 = embedding_fn(first, second)

        dist = np.linalg.norm(result1 - result2)
        dist_list.append(dist)
        if idx % 1000 == 0:
            print('complete %d pairs' % idx)
    print('cost_times: %.2f sec' % (timeit.default_timer() - start))

    thresholds = np.arange(0.1, 3.0, 0.05)

    accs = []
    tps = []
    fps = []
    fns = []
    tns = []
    for threshold in thresholds:
        pred_same = np.less(dist_list, threshold)
        tp = np.sum(np.logical_and(pred_same, true_same))
        tn = np.sum(np.logical_and(np.logical_not(pred_same), np.logical_not(true_same)))
        fp = diff - tn
        fn = same - tp
        acc = float(tp + tn) / total
        accs.append(acc)
        tps.append(int(tp))
        tns.append(int(tn))
        fps.append(int(fp))
        fns.append(int(fn))
    print('thresholds:', ", ".join("%.2f" % f for f in thresholds))
    print('accs:', ", ".join("%.2f" % f for f in accs))
    tpr = [0 if (tp + fn == 0) else float(tp) / float(tp + fn) for tp, fn in zip(tps, fns)]
    fpr = [0 if (fp + tn == 0) else float(fp) / float(fp + tn) for fp, tn in zip(fps, tns)]
    best_index = int(np.argmax(accs))

    return accs[best_index], thresholds[best_index], tps[best_index], fps[best_index], fns[best_index], tns[
        best_index], tpr, fpr


def ver_test(data_set, sess, l2_embedding_tensor, input_placeholder):
    first_list, second_list, true_same = data_set[0], data_set[1], np.array(data_set[2])

    dist_list = []
    for first, second in zip(first_list, second_list):
        vector_pair = sess.run(l2_embedding_tensor, feed_dict={input_placeholder: np.stack((first, second), axis=0)})
        dist = np.linalg.norm(vector_pair[0] - vector_pair[1])
        dist_list.append(dist)

    thresholds = np.arange(0.1, 3.0, 0.1)

    accs = []
    for threshold in thresholds:
        pred_same = np.less(dist_list, threshold)
        tp = np.sum(np.logical_and(pred_same, true_same))
        tn = np.sum(np.logical_and(np.logical_not(pred_same), np.logical_not(true_same)))
        acc = float(tp + tn) / len(first_list)
        accs.append(acc)
    best_threshold_index = int(np.argmax(accs))

    return accs[best_threshold_index], thresholds[best_threshold_index]


def plot_roc(fpr, tpr):
    import matplotlib.pyplot as plt
    plt.figure()
    lw = 2
    plt.plot(fpr, tpr, color='darkorange',
             lw=lw, label='ROC curve')
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic')
    plt.legend(loc="lower right")
    plt.show()


def load_bin(bin_path, input_size):
    import mxnet as mx

    bins, issame_list = pickle.load(open(bin_path, 'rb'), encoding='bytes')
    first_imgs = np.empty((len(issame_list) * 2, input_size[0], input_size[1], 3))
    second_imgs = np.empty((len(issame_list) * 2, input_size[0], input_size[1], 3))

    for i in range(len(issame_list) * 2):
        _bin = bins[i]
        img = mx.image.imdecode(_bin).asnumpy()  # BGR
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, input_size)
        img = img - 127.5
        img = img * 0.0078125
        for flip in [0, 1]:
            if flip == 1:
                img = np.fliplr(img)
            if i % 2 == 0:
                first_imgs[i + flip, ...] = img
            else:
                second_imgs[i - 1 + flip, ...] = img
        i += 1
        if i % 1000 == 0:
            print('loading bin', i)

    return first_imgs, second_imgs, np.repeat(issame_list, 2)


def test_tfrecord(tfrecord, embedding_fn, shape, is_plot=False):
    verification_path = os.path.join('tfrecord', tfrecord)
    ver_dataset = get_ver_data(verification_path, shape)

    val_acc, val_thr, tp, fp, fn, tn, tpr, fpr = ver_tfrecord(ver_dataset, embedding_fn)
    print('test accuracy is: %.3f, thr: %.2f, prec: %.3f, rec: %.3f.' %
          (val_acc, val_thr, float(tp) / (tp + fp), float(tp) / (tp + fn)))

    if is_plot:
        plot_roc(fpr, tpr)


def test_lfw(path, embedding_fn, shape, is_plot=False):
    ver_dataset = load_bin(path, shape)

    val_acc, val_thr, tp, fp, fn, tn, tpr, fpr = ver_tfrecord(ver_dataset, embedding_fn)
    print('test accuracy is: %.3f, thr: %.2f, prec: %.3f, rec: %.3f.' %
          (val_acc, val_thr, float(tp) / (tp + fp), float(tp) / (tp + fn)))

    if is_plot:
        plot_roc(fpr, tpr)
