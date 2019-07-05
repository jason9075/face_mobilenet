import cv2
import numpy as np
import tensorflow as tf
from sklearn import preprocessing


def parse_function(example_proto):
    features = {'image_raw': tf.FixedLenFeature([], tf.string),
                'label': tf.FixedLenFeature([], tf.int64)}
    features = tf.parse_single_example(example_proto, features)
    # You can do more image distortion here for training data
    img = tf.image.decode_jpeg(features['image_raw'])
    img = tf_pre_process_image(img)
    img = tf.image.random_flip_left_right(img)
    label = tf.cast(features['label'], tf.int64)
    return img, label


def tf_pre_process_image(img):
    img = tf.reshape(img, shape=(112, 112, 3))
    r, g, b = tf.split(img, num_or_size_splits=3, axis=-1)
    img = tf.concat([b, g, r], axis=-1)
    img = tf.cast(img, dtype=tf.float32)
    img = tf.subtract(img, 127.5)
    img = tf.multiply(img, 0.0078125)
    return img


def pre_process_image(img):
    img = np.array(img, dtype=np.float32)
    img -= 127.5
    img *= 0.0078125
    return img


def get_ver_data(record_path):
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
        img_first = pre_process_image(img_first)
        first_list.append(img_first)

        image_string = example.features.feature['image_second'].bytes_list.value[0]
        img = np.fromstring(image_string, dtype=np.uint8)
        img_second = cv2.imdecode(img, cv2.IMREAD_COLOR)
        img_second = pre_process_image(img_second)
        second_list.append(img_second)

        is_same = example.features.feature['is_same'].int64_list.value[0]
        is_same_list.append(is_same)

    return [first_list, second_list, is_same_list]


def ver_test(data_set, sess, embedding_tensor, feed_dict, input_placeholder):

    first_list, second_list, true_same = data_set[0], data_set[1], np.array(data_set[2])

    dist_list = []
    for first, second in zip(first_list, second_list):
        feed_dict[input_placeholder] = np.stack((first, second), axis=0)
        vector_pair = sess.run(embedding_tensor, feed_dict)
        vector_pair = preprocessing.normalize([vector_pair]).flatten()
        dist = np.linalg.norm(vector_pair[0] - vector_pair[1])
        dist_list.append(dist)

    thresholds = np.arange(0.1, 4, 0.1)

    accs = []
    for threshold in thresholds:
        pred_same = np.less(dist_list, threshold)
        tp = np.sum(np.logical_and(pred_same, true_same))
        tn = np.sum(np.logical_and(np.logical_not(pred_same), np.logical_not(true_same)))
        acc = float(tp + tn) / len(first_list)
        accs.append(acc)
    best_threshold_index = int(np.argmax(accs))

    return accs[best_threshold_index], thresholds[best_threshold_index]
