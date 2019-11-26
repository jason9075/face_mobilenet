import cv2
import numpy as np
import sklearn
import tensorflow as tf

import utils

CKPT_NAME = 'InsightFace_iter_18000.ckpt'
SHAPE = (224, 224)


def main():
    img_1 = cv2.imread('images/astra_door_align/betty/2019-06-12_16-44-44_ivy_375_78_73_dc93d3_28.jpg')
    img_1 = cv2.cvtColor(img_1, cv2.COLOR_BGR2RGB)
    img_1 = utils.pre_process_image(img_1, shape=SHAPE)

    img_2 = cv2.imread('images/astra_door_align/jason/2019-06-12_15-42-15_jason_349_146_113_941e14_45.jpg')
    img_2 = cv2.cvtColor(img_2, cv2.COLOR_BGR2RGB)
    img_2 = utils.pre_process_image(img_2, shape=SHAPE)

    with tf.Session() as sess:
        saver = tf.train.import_meta_graph(
            f'model_out/{CKPT_NAME}.meta', clear_devices=True)
        saver.restore(sess, f"model_out/{CKPT_NAME}")

        input_tensor = tf.get_default_graph().get_tensor_by_name(
            "input_images:0")
        embedding_tensor = tf.get_default_graph().get_tensor_by_name(
            "gdc/embedding/Identity:0")

        f1 = sess.run(embedding_tensor, feed_dict={input_tensor: np.expand_dims(img_1, axis=0)})[0]
        f1 = sklearn.preprocessing.normalize(np.expand_dims(f1, 0)).flatten()
        print('--->', f1)

        f2 = sess.run(embedding_tensor, feed_dict={input_tensor: np.expand_dims(img_2, axis=0)})[0]
        f2 = sklearn.preprocessing.normalize(np.expand_dims(f2, 0)).flatten()
        print('--->', f2)

        dist = np.sum(np.square(f1 - f2))
        sim = np.dot(f1, f2.T)
        print(f'dist: {dist}')
        print(f'sim: {sim}')


if __name__ == '__main__':
    main()
