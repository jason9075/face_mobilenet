import tensorflow as tf
import efficientnet.tfkeras as efn
import tensorflow_addons as tfa

from tensorflow.keras.models import load_model
from tensorflow.keras.layers import Lambda, Dense, GlobalAveragePooling2D

EMB_SIZE = 128
IMG_SHAPE = (112, 112, 3)


def main():
    # model = load_model('model_out/keras_best_embedding')
    # model.save_weights('model_out/tmp.h5')

    # base_model = tf.keras.applications.ResNet50V2(input_shape=IMG_SHAPE, include_top=False, weights='imagenet')
    base_model = efn.EfficientNetB2(input_shape=IMG_SHAPE,
                                    include_top=False, weights='imagenet')

    model = tf.keras.Sequential([
        base_model,
        GlobalAveragePooling2D(),
        Dense(EMB_SIZE, name='embedding', use_bias=False),
        Lambda(lambda x: tf.math.l2_normalize(x, axis=1), name='l2_embedding')
    ])
    model.load_weights('model_out/20-0.75.h5', by_name=True)
    model.summary()

    model.save('model_out/inference', include_optimizer=False)


if __name__ == '__main__':
    main()
