import cv2
import numpy as np
import tensorflow as tf
from sklearn import preprocessing

import utils

SHAPE = (224, 224)
IMG_SHAPE = (224, 224, 3)
EMB_SIZE = 128
CLASS_NUM = 1036
PLOT_ROC = True
LFW_PATH = 'tfrecord/lfw.bin'
RESTORE_BY_WEIGHT = True


def main():
    if RESTORE_BY_WEIGHT:
        # model = tf.keras.models.load_model('model_out/keras_best')
        # model.save_weights('model_out/keras_best/weight.h5')
        model = restore_weight('model_out/keras_best/weight.h5')
    else:
        model = tf.keras.models.load_model('model_out/keras_embedding')

    def embedding_fn(img1, img2):
        h, w, _ = img1.shape
        result1 = model.predict(np.expand_dims(img1, axis=0))
        result2 = model.predict(np.expand_dims(img2, axis=0))
        result1 = preprocessing.normalize(result1, norm='l2')
        result2 = preprocessing.normalize(result2, norm='l2')

        return result1, result2

    # gen_model()
    utils.test_tfrecord('verification.tfrecord', embedding_fn, SHAPE, is_plot=PLOT_ROC)
    # utils.test_lfw(LFW_PATH, embedding_fn, SHAPE, is_plot=PLOT_ROC)


def restore_weight(path):
    base_model = tf.keras.applications.ResNet50V2(input_shape=IMG_SHAPE, include_top=False, weights='imagenet')

    model = tf.keras.Sequential([
        base_model,
        tf.keras.layers.GlobalAveragePooling2D(),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(EMB_SIZE),
    ])

    model.load_weights(path, by_name=True)

    return model


def pre_processing(path):
    img = cv2.imread(path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img - 127.5
    img = img * 0.0078125
    return img


def single_test():
    model = tf.keras.models.load_model('model_out/keras_embedding')

    img1 = pre_processing('images/star224/Arak Amornsuppasiri/000022_0.jpg')
    img2 = pre_processing('images/star224/Arak Amornsuppasiri/000003_0.jpg')

    result1 = model.predict(np.expand_dims(img1, axis=0))
    result2 = model.predict(np.expand_dims(img2, axis=0))
    print(np.argmax(result1[0]))
    print(np.argmax(result2[0]))

    emb1 = preprocessing.normalize(result1, norm='l2')
    emb2 = preprocessing.normalize(result2, norm='l2')

    print(np.linalg.norm(emb1 - emb2))


if __name__ == '__main__':
    main()
