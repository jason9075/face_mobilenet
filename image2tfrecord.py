import glob
import random

import cv2
import os
import pickle
import mxnet as mx
import tensorflow as tf
import numpy as np

KEY_IMAGE = 'image_raw'
KEY_LABEL = 'label'
KEY_TEXT = 'text'
IMAGE_SIZE = (112, 112)
SAME_PER_PERSON = 20


def gen_train_tfrecord():
    output_path = os.path.join('tfrecord', 'train.tfrecord')
    writer = tf.python_io.TFRecordWriter(output_path)

    directory = os.path.join('images', 'image_db')
    faces = [
        o for o in os.listdir(directory) if os.path.isdir(os.path.join(directory, o))
    ]
    faces = {f: glob.glob(os.path.join(directory, f, '*.jpg')) for f in faces}

    write_record(writer, faces)
    writer.close()


def show_bin_image():
    bins, issame_list = pickle.load(open(os.path.join('images', 'vgg2_fp.bin'), 'rb'), encoding='bytes')
    for i in range(len(bins)):
        img_info = bins[i]
        img = mx.image.imdecode(img_info).asnumpy()
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        cv2.imshow('frame', img)
        cv2.waitKey(0)


def show_train_tfrecord_image():
    input_path = os.path.join('tfrecord', 'train.tfrecord')
    record_iterator = tf.python_io.tf_record_iterator(path=input_path)
    for record in record_iterator:
        example = tf.train.Example()
        example.ParseFromString(record)
        image_string = example.features.feature[KEY_IMAGE].bytes_list.value[0]
        img = np.fromstring(image_string, dtype=np.uint8)
        img = cv2.imdecode(img, cv2.IMREAD_COLOR)
        label = example.features.feature[KEY_LABEL].int64_list.value[0]
        text = example.features.feature[KEY_TEXT].bytes_list.value[0]

        print(f'label:{label}, text:{text}')
        cv2.imshow('frame', img)
        cv2.waitKey(0)


def write_record(writer, faces):
    for i, (k, paths) in enumerate(faces.items()):
        text = k.encode('utf8')
        label = i
        for path in paths:
            img = cv2.imread(path)
            img = cv2.resize(img, IMAGE_SIZE)
            img = cv2.imencode('.jpg', img)[1].tostring()
            example = tf.train.Example(features=tf.train.Features(feature={
                KEY_IMAGE: tf.train.Feature(bytes_list=tf.train.BytesList(value=[img])),
                KEY_LABEL: tf.train.Feature(int64_list=tf.train.Int64List(value=[label])),
                KEY_TEXT: tf.train.Feature(bytes_list=tf.train.BytesList(value=[text]))
            }))
            writer.write(example.SerializeToString())  # Serialize To String
        if i % 10 == 0:
            print('%d person processed' % i)


def write_ver_record(writer, faces):
    for i, (k, _) in enumerate(faces.items()):

        same_pool = [path for (key, paths) in faces.items() if key == k for path in paths]
        diff_pool = [path for (key, paths) in faces.items() if key != k for path in paths]

        print(np.random.choice(same_pool, size=(12, 2), replace=False))
        # for path in paths:
        #     img = cv2.imread(path)
        #     img = cv2.resize(img, IMAGE_SIZE)
        #     img = cv2.imencode('.jpg', img)[1].tostring()
        #     example = tf.train.Example(features=tf.train.Features(feature={
        #         KEY_IMAGE: tf.train.Feature(bytes_list=tf.train.BytesList(value=[img])),
        #         KEY_LABEL: tf.train.Feature(int64_list=tf.train.Int64List(value=[label])),
        #         KEY_TEXT: tf.train.Feature(bytes_list=tf.train.BytesList(value=[text]))
        #     }))
        #     writer.write(example.SerializeToString())  # Serialize To String
        # if i % 10 == 0:
        #     print('%d person processed' % i)
    print('asdasd')


def load_bin(db_name, image_size):
    bins, issame_list = pickle.load(open(os.path.join('images', db_name + '.bin'), 'rb'), encoding='bytes')
    data_list = []
    for _ in [0, 1]:
        data = np.empty((len(issame_list) * 2, image_size[0], image_size[1], 3))
        data_list.append(data)
    for i in range(len(issame_list) * 2):
        _bin = bins[i]
        img = mx.image.imdecode(_bin).asnumpy()
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        for flip in [0, 1]:
            if flip == 1:
                img = np.fliplr(img)
            data_list[flip][i, ...] = img
        i += 1
        if i % 1000 == 0:
            print('loading bin', i)
    print(data_list[0].shape)
    return data_list, issame_list


def gen_verification_tfrecord():
    output_path = os.path.join('tfrecord', 'train.tfrecord')
    writer = tf.python_io.TFRecordWriter(output_path)

    directory = os.path.join('images', 'image_db')
    faces = [
        o for o in os.listdir(directory) if os.path.isdir(os.path.join(directory, o))
    ]

    faces = {f: glob.glob(os.path.join(directory, f, '*.jpg')) for f in faces}

    write_ver_record(writer, faces)
    # writer.close()


if __name__ == '__main__':
    # gen_train_tfrecord()
    # show_train_tfrecord_image()
    gen_verification_tfrecord()
