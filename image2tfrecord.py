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
KEY_IS_SAME = 'is_same'
KEY_IMAGE_FIRST = 'image_first'
KEY_IMAGE_SECOND = 'image_second'
KEY_FIRST_NAME = 'first_name'
KEY_SECOND_NAME = 'second_name'
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

        same_pairs = np.random.choice(same_pool, size=(len(same_pool)//2, 2), replace=False)
        diff_face = np.random.choice(diff_pool, size=len(same_pool), replace=False)
        diff_pairs = np.vstack((same_pool, list(diff_face))).transpose()

        write_pair(same_pairs, writer, is_same=1)
        write_pair(diff_pairs, writer, is_same=0)


def write_pair(same_pairs, writer, is_same):
    for same1, same2 in same_pairs:
        print(f'{same1} vs {same2} is {is_same}')
        img1 = cv2.imread(same1)
        img1 = cv2.resize(img1, IMAGE_SIZE)
        img1 = cv2.imencode('.jpg', img1)[1].tostring()
        img2 = cv2.imread(same2)
        img2 = cv2.resize(img2, IMAGE_SIZE)
        img2 = cv2.imencode('.jpg', img2)[1].tostring()
        first_name = same1.split('/')[-1].encode('utf8')
        second_name = same2.split('/')[-1].encode('utf8')

        example = tf.train.Example(features=tf.train.Features(feature={
            KEY_IMAGE_FIRST: tf.train.Feature(bytes_list=tf.train.BytesList(value=[img1])),
            KEY_IMAGE_SECOND: tf.train.Feature(bytes_list=tf.train.BytesList(value=[img2])),
            KEY_FIRST_NAME: tf.train.Feature(bytes_list=tf.train.BytesList(value=[first_name])),
            KEY_SECOND_NAME: tf.train.Feature(bytes_list=tf.train.BytesList(value=[second_name])),
            KEY_IS_SAME: tf.train.Feature(int64_list=tf.train.Int64List(value=[is_same]))
        }))
        writer.write(example.SerializeToString())


def gen_verification_tfrecord():
    output_path = os.path.join('tfrecord', 'verification.tfrecord')
    writer = tf.python_io.TFRecordWriter(output_path)

    directory = os.path.join('images', 'astra_door_align')
    faces = [
        o for o in os.listdir(directory) if os.path.isdir(os.path.join(directory, o))
    ]

    faces = {f: glob.glob(os.path.join(directory, f, '*.jpg')) for f in faces}

    write_ver_record(writer, faces)
    writer.close()


if __name__ == '__main__':
    # gen_train_tfrecord()
    # show_train_tfrecord_image()
    gen_verification_tfrecord()
