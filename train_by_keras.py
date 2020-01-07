import os
import pathlib

import numpy as np
import tensorflow as tf
from sklearn import preprocessing

import utils

tf.random.set_seed(9075)
IMG_SHAPE = (224, 224, 3)
SHAPE = (224, 224)
BATCH_SIZE = 64
AUTOTUNE = tf.data.experimental.AUTOTUNE
CLASS_NAMES = np.array([])
EPOCHS = 30000
TRAIN_DATA_PATH = 'images/public_face_1036_224_train/'
# TRAIN_DATA_PATH = 'images/star224/'
TEST_DATA_PATH = 'images/public_face_1036_224_valid/'
# TEST_DATA_PATH = 'images/star224/'
OUTPUT_MODEL_FOLDER = 'model_out/keras'
OUTPUT_MODEL_LOGS_FOLDER = 'model_out/keras_logs'
OUTPUT_MODEL_FOLDER_CKPT = 'model_out/keras_ckpt'
OUTPUT_EMB_MODEL_FOLDER = 'model_out/keras_embedding'
PATIENCE = 100
EMB_SIZE = 128


class SaveBestValCallback(tf.keras.callbacks.Callback):

    def __init__(self):
        super().__init__()
        self.embedding_model = None
        self.best_acc = 0

    def set_model(self, model):
        print('set_model')
        self.model = model
        embedding = model.layers[4].output
        self.embedding_model = tf.keras.models.Model(inputs=model.input, outputs=embedding)

    def on_epoch_end(self, epoch, logs=None):
        verification_path = os.path.join('tfrecord', 'verification.tfrecord')
        ver_dataset = utils.get_ver_data(verification_path, SHAPE)

        def embedding_fn(img1, img2):
            h, w, _ = img1.shape
            result1 = self.embedding_model.predict(np.expand_dims(img1, axis=0))
            result2 = self.embedding_model.predict(np.expand_dims(img2, axis=0))
            result1 = preprocessing.normalize(result1, norm='l2')
            result2 = preprocessing.normalize(result2, norm='l2')

            return result1, result2

        val_acc, val_thr, _, _, _, _, _, _ = utils.ver_tfrecord(ver_dataset, embedding_fn, verbose=True)
        print('val_acc: %f, val_thr: %f ' % (val_acc, val_thr))
        if self.best_acc < val_acc:
            print('best_acc < val_acc | %f < %f ' % (self.best_acc, val_acc))
            self.embedding_model.save(OUTPUT_EMB_MODEL_FOLDER)
            self.best_acc = val_acc


def get_label(file_path):
    parts = tf.strings.split(file_path, '/')
    one_hot = tf.cast(parts[-2] == CLASS_NAMES, tf.int8)
    return tf.argmax(one_hot)


def decode_img(img):
    img = tf.image.decode_jpeg(img, channels=3)  # value from 0 ~ 1
    img = tf.image.convert_image_dtype(img, tf.float32)
    img = tf.image.resize(img, [IMG_SHAPE[0], IMG_SHAPE[1]])
    img = tf.image.random_brightness(img, 0.2)
    img = tf.image.random_saturation(img, 0.6, 1.6)
    img = tf.image.random_contrast(img, 0.6, 1.4)
    img = tf.image.random_flip_left_right(img)
    img = tf.subtract(img, 0.5)
    img = tf.multiply(img, 2)
    return img


def process_path(file_path):
    label = get_label(file_path)
    img = tf.io.read_file(file_path)
    img = decode_img(img)
    return img, label


def prepare_for_training(ds, cache=False, shuffle_buffer_size=2000):
    if cache:
        if isinstance(cache, str):
            ds = ds.cache(cache)
        else:
            ds = ds.cache()

    ds = ds.shuffle(buffer_size=shuffle_buffer_size)
    ds = ds.repeat()
    ds = ds.batch(BATCH_SIZE)
    ds = ds.prefetch(buffer_size=AUTOTUNE)

    return ds


def main():
    global CLASS_NAMES

    train_data_dir = pathlib.Path(TRAIN_DATA_PATH)
    list_ds = tf.data.Dataset.list_files(str(train_data_dir / '*/*.jpg'))
    CLASS_NAMES = np.array([item.name for item in train_data_dir.glob('*') if item.name not in [".keep", ".DS_Store"]])
    test_data_dir = pathlib.Path(TEST_DATA_PATH)
    test_list_ds = tf.data.Dataset.list_files(str(test_data_dir / '*/*.jpg'))
    train_image_count = len(list(train_data_dir.glob('*/*.jpg')))
    test_image_count = len(list(test_data_dir.glob('*/*.jpg')))
    steps_per_epoch = np.ceil(train_image_count / BATCH_SIZE)
    val_steps = np.ceil(test_image_count / BATCH_SIZE)

    print('total labels: %d' % len(CLASS_NAMES))

    train_labeled_ds = list_ds.map(process_path, num_parallel_calls=AUTOTUNE)
    train_ds = prepare_for_training(train_labeled_ds)
    test_labeled_ds = test_list_ds.map(process_path, num_parallel_calls=AUTOTUNE)

    base_model = tf.keras.applications.ResNet50V2(input_shape=IMG_SHAPE, include_top=False, weights='imagenet')

    model = tf.keras.Sequential([
        base_model,
        tf.keras.layers.GlobalAveragePooling2D(),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(EMB_SIZE),
        # tf.keras.layers.Activation(tf.nn.relu6),
        tf.keras.layers.Dense(len(CLASS_NAMES), activation='softmax')
    ])

    model.summary()

    model.compile(optimizer=tf.keras.optimizers.Adam(),
                  loss='sparse_categorical_crossentropy',
                  metrics=['sparse_categorical_accuracy'])

    save_cb = tf.keras.callbacks.ModelCheckpoint(OUTPUT_MODEL_FOLDER_CKPT, monitor='val_loss', verbose=1,
                                                 save_weights_only=False, mode='auto')

    summary_cb = tf.keras.callbacks.TensorBoard(OUTPUT_MODEL_LOGS_FOLDER, histogram_freq=1)

    model.fit(train_ds,
              epochs=EPOCHS,
              steps_per_epoch=steps_per_epoch,
              validation_data=test_labeled_ds.batch(BATCH_SIZE),
              validation_steps=val_steps,
              callbacks=[SaveBestValCallback(), save_cb, summary_cb])

    loss, accuracy = model.evaluate(test_labeled_ds.batch(BATCH_SIZE), verbose=2)
    print("Loss :", loss)
    print("Accuracy :", accuracy)


class EarlyStoppingAtMinLoss(tf.keras.callbacks.Callback):
    def __init__(self):
        super(EarlyStoppingAtMinLoss, self).__init__()
        self.patience = PATIENCE
        self.best_weights = None
        self.stopped_epoch = None
        self.wait = 0
        self.stopped_epoch = 0
        self.best = np.Inf

    def on_epoch_end(self, epoch, logs=None):
        current = logs.get('val_loss')
        if current is None:
            print('val_loss is None.')
            return
        if np.less(current, self.best):
            self.best = current
            self.wait = 0
            # Record the best weights if current results is better (less).
            self.best_weights = self.model.get_weights()
        else:
            self.wait += 1
            if self.patience < self.wait:
                self.stopped_epoch = epoch
                self.model.stop_training = True
                self.model.set_weights(self.best_weights)

    def on_train_end(self, logs=None):
        if 0 < self.stopped_epoch:
            print('Restoring model weights from the end of the best epoch. val_loss: %03f:' % self.best)
            print('Epoch %05d: early stopping' % (self.stopped_epoch + 1))


if __name__ == '__main__':
    main()
