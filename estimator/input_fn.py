import os
import tensorflow as tf

import utils


def train_input_fn(tfrecord_name, params):
    record_path = os.path.join('tfrecord', tfrecord_name)
    data_set = tf.data.TFRecordDataset(record_path)
    data_set = data_set.map(utils.parse_function)
    data_set = data_set.shuffle(buffer_size=params.buffer_size)
    data_set = data_set.batch(params.batch_size)
    data_set = data_set.repeat()

    iterator = data_set.make_one_shot_iterator()
    images_train, labels_train = iterator.get_next()
    return images_train, labels_train


def test_input_fn(tfrecord_name, params):
    record_path = os.path.join('tfrecord', tfrecord_name)
    data_set = tf.data.TFRecordDataset(record_path)
    data_set = data_set.map(utils.parse_function)
    data_set = data_set.batch(params.batch_size)

    iterator = data_set.make_one_shot_iterator()
    images_train, labels_train = iterator.get_next()
    return images_train, labels_train
