import efficientnet.tfkeras as efn
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Lambda, Dense, GlobalAveragePooling2D

import utils
from test_by_keras_model import pre_processing

EMB_SIZE = 128
IMG_SHAPE = (112, 112, 3)
SHAPE = (112, 112)
H5_PATH = 'model_out/20-0.75.h5'
VER_NAME = 'public_face_1036_112_align_verification.tfrecord'
VER_TYPE = 'euclidean'  # euclidean, cosine


def single():
    base_model = efn.EfficientNetB2(input_shape=IMG_SHAPE,
                                    include_top=False, weights='imagenet')

    model = tf.keras.Sequential([
        base_model,
        GlobalAveragePooling2D(),
        Dense(EMB_SIZE, name='embedding', use_bias=False),
        Lambda(lambda x: tf.math.l2_normalize(x, axis=1), name='l2_embedding')
    ])
    model.load_weights(H5_PATH, by_name=True)
    # model.summary()

    img1 = pre_processing('images/public_face_1036_112_train_align/三枝夕下/000004_0.jpg')
    # img2 = pre_processing('images/train_image/0_2924088/2.jpg')
    img2 = pre_processing('images/public_face_1036_112_train_align/上川隆也/000002_0.jpg')

    result1 = model.predict(np.expand_dims(img1, axis=0))
    result2 = model.predict(np.expand_dims(img2, axis=0))

    val = np.linalg.norm(result1 - result2)
    print(val)


def batch():
    base_model = efn.EfficientNetB2(input_shape=IMG_SHAPE,
                                    include_top=False, weights='imagenet')

    model = tf.keras.Sequential([
        base_model,
        GlobalAveragePooling2D(),
        Dense(EMB_SIZE, name='embedding', use_bias=False),
        Lambda(lambda x: tf.math.l2_normalize(x, axis=1), name='l2_embedding')
    ])
    model.load_weights(H5_PATH, by_name=True)

    def embedding_fn(img1, img2):
        h, w, _ = img1.shape
        result1 = model.predict(np.expand_dims(img1, axis=0))
        result2 = model.predict(np.expand_dims(img2, axis=0))

        return result1, result2

    utils.test_tfrecord(VER_NAME,
                        embedding_fn,
                        SHAPE,
                        is_plot=True,
                        verbose=True,
                        ver_type=VER_TYPE)


if __name__ == '__main__':
    # single()
    batch()
