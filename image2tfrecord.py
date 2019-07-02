import glob

import cv2
import os
import pickle
import mxnet as mx
import tensorflow as tf
import numpy as np

KEY_IMAGE = 'image_raw'
KEY_LABEL = 'label'
KEY_TEXT = 'text'
IMAGE_SIZE = (224, 224)


def main():
    output_path = os.path.join('tfrecord', 'train.tfrecord')
    writer = tf.python_io.TFRecordWriter(output_path)

    directory = os.path.join('images', 'image_db')
    faces = [
        o for o in os.listdir(directory) if os.path.isdir(os.path.join(directory, o))
    ]
    faces = {f: glob.glob(os.path.join(directory, f, '*.jpg')) for f in faces}

    write_record(writer, faces)


def show_bin_image():
    bins, issame_list = pickle.load(open(os.path.join('images', 'vgg2_fp.bin'), 'rb'), encoding='bytes')
    for i in range(len(bins)):
        img_info = bins[i]
        img = mx.image.imdecode(img_info).asnumpy()
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        cv2.imshow('frame', img)
        cv2.waitKey(0)


def show_tfrecord_image():
    input_path = os.path.join('tfrecord', 'train.tfrecord')
    record_iterator = tf.python_io.tf_record_iterator(path=input_path)
    for record in record_iterator:
        example = tf.train.Example()
        example.ParseFromString(record)
        image_string = example.features.feature[KEY_IMAGE].bytes_list.value[0]
        img = np.fromstring(image_string, dtype=np.uint8)
        img = cv2.imdecode(img, cv2.IMREAD_COLOR)
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
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
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.imencode('.jpg', img)[1]
            img = np.array(img).tostring()
            example = tf.train.Example(features=tf.train.Features(feature={
                KEY_IMAGE: tf.train.Feature(bytes_list=tf.train.BytesList(value=[img])),
                KEY_LABEL: tf.train.Feature(int64_list=tf.train.Int64List(value=[label])),
                KEY_TEXT: tf.train.Feature(bytes_list=tf.train.BytesList(value=[text]))
            }))
            writer.write(example.SerializeToString())  # Serialize To String
        if i % 10 == 0:
            print('%d person processed' % i)
    writer.close()


if __name__ == '__main__':
    # main()
    show_tfrecord_image()
