import datetime

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
    img = tf_pre_process_image(img)
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


def lfw_test(data_set, sess, embedding_tensor, feed_dict, input_placeholder):
    def data_iter(datasets, batch_size):
        data_num = datasets.shape[0]
        for i in range(0, data_num, batch_size):
            yield datasets[i:min(i + batch_size, data_num), ...]

    batch_size = 32
    data_list = data_set[0]
    issame_list = data_set[1]
    embeddings_list = []
    time_consumed = 0.0
    for i in range(len(data_list)):
        datas = data_list[i]
        embeddings = None
        feed_dict.setdefault(input_placeholder, None)
        for idx, data in enumerate(data_iter(datas, batch_size)):
            data_tmp = data.copy()  # fix issues #4
            data_tmp -= 127.5
            data_tmp *= 0.0078125
            feed_dict[input_placeholder] = data_tmp
            time0 = datetime.datetime.now()
            _embeddings = sess.run(embedding_tensor, feed_dict)
            time_now = datetime.datetime.now()
            diff = time_now - time0
            time_consumed += diff.total_seconds()
            if embeddings is None:
                embeddings = np.zeros((datas.shape[0], _embeddings.shape[1]))
            try:
                embeddings[idx * batch_size:min((idx + 1) * batch_size, datas.shape[0]), ...] = _embeddings
            except ValueError:
                print(
                    'idx*batch_size value is %d min((idx+1)*batch_size, datas.shape[0]) %d, batch_size %d, data.shape[0] %d' %
                    (idx * batch_size, min((idx + 1) * batch_size, datas.shape[0]), batch_size, datas.shape[0]))
                print('embedding shape is ', _embeddings.shape)
        embeddings_list.append(embeddings)

    embeddings = embeddings_list[0] + embeddings_list[1]
    embeddings = sklearn.preprocessing.normalize(embeddings)

    thresholds = np.arange(0.1, 4.0, 0.1)
    embeddings1 = embeddings[0::2]
    embeddings2 = embeddings[1::2]

    dist_list = []
    for first, second in zip(embeddings1, embeddings2):
        dist = np.linalg.norm(first - second)
        dist_list.append(dist)

    accs = []
    for threshold in thresholds:
        pred_same = np.less(dist_list, threshold)
        tp = np.sum(np.logical_and(pred_same, issame_list))
        tn = np.sum(np.logical_and(np.logical_not(pred_same), np.logical_not(issame_list)))
        acc = float(tp + tn) / len(issame_list)
        accs.append(acc)
        print('threshold: %.2f, acc: %.2f' % (threshold, acc))
    best_threshold_index = int(np.argmax(accs))

    return accs[best_threshold_index], thresholds[best_threshold_index]
