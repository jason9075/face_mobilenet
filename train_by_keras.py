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
TRAIN_DATA_PATH = 'images/public_face_1036_224/'
# TRAIN_DATA_PATH = 'images/glint_tiny/'
OUTPUT_MODEL_LOGS_FOLDER = 'model_out/keras_logs'
OUTPUT_EMB_MODEL_FOLDER = 'model_out/keras_embedding'
OUTPUT_BEST_EMB_MODEL_FOLDER = 'model_out/keras_best_embedding'
VER_RECORD = 'public_face_1036_112_align_verification.tfrecord'
EMB_SIZE = 128


class SaveBestValCallback(tf.keras.callbacks.Callback):

    def __init__(self):
        super().__init__()
        self.embedding_model = None
        self.best_acc = 0

    def set_model(self, model):
        self.model = model
        embedding = model.get_layer(name='l2_embedding')
        self.embedding_model = tf.keras.models.Model(inputs=model.input, outputs=embedding.output)

    def on_epoch_end(self, epoch, logs=None):
        verification_path = os.path.join('tfrecord', VER_RECORD)
        ver_dataset = utils.get_ver_data(verification_path, SHAPE)

        first_list, second_list, true_same = ver_dataset[0], ver_dataset[1], np.array(ver_dataset[2])
        total = len(true_same)

        sim_list = []
        for idx, (img1, img2) in enumerate(zip(first_list, second_list)):
            h, w, _ = img1.shape
            result1 = self.embedding_model.predict(np.expand_dims(img1, axis=0))
            result2 = self.embedding_model.predict(np.expand_dims(img2, axis=0))
            # result1 = preprocessing.normalize(result1, norm='l2')
            # result2 = preprocessing.normalize(result2, norm='l2')

            sim = np.dot(result1, np.transpose(result2)) / (np.sqrt(np.dot(result1, np.transpose(result1))) * np.sqrt(
                np.dot(result2, np.transpose(result2))))
            sim_list.append(sim[0][0])

        thresholds = np.arange(0.1, 3.0, 0.05)

        accs = []
        for threshold in thresholds:
            pred_same = np.less(sim_list, threshold)
            tp = np.sum(np.logical_and(pred_same, true_same))
            tn = np.sum(np.logical_and(np.logical_not(pred_same), np.logical_not(true_same)))
            acc = float(tp + tn) / total
            accs.append(acc)
        best_index = int(np.argmax(accs))

        val_acc = accs[best_index]
        val_thr = thresholds[best_index]

        print('\n val_acc: %f, val_thr: %f, current best: %f ' % (val_acc, val_thr, self.best_acc))
        if self.best_acc < val_acc:
            self.embedding_model.save(OUTPUT_BEST_EMB_MODEL_FOLDER)
            self.best_acc = val_acc
            return
        self.embedding_model.save(OUTPUT_EMB_MODEL_FOLDER)


def get_label(file_path):
    parts = tf.strings.split(file_path, '/')
    one_hot = tf.cast(parts[-2] == CLASS_NAMES, tf.int8)
    return tf.argmax(one_hot)


def decode_img(img):
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.random_brightness(img, 0.2)
    img = tf.image.random_saturation(img, 0.6, 1.6)
    img = tf.image.random_contrast(img, 0.6, 1.4)
    img = tf.image.random_flip_left_right(img)
    img = tf.image.convert_image_dtype(img, tf.float32)
    img = tf.subtract(img, 0.5)
    img = tf.multiply(img, 2.0)
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
    train_image_count = len(list(train_data_dir.glob('*/*.jpg')))
    steps_per_epoch = np.ceil(train_image_count / BATCH_SIZE)

    print('total labels: %d' % len(CLASS_NAMES))

    train_labeled_ds = list_ds.map(process_path, num_parallel_calls=AUTOTUNE)
    train_ds = prepare_for_training(train_labeled_ds)

    base_model = tf.keras.applications.ResNet50V2(input_shape=IMG_SHAPE, include_top=False, weights='imagenet')
    base_model.summary()

    model = tf.keras.Sequential([
        base_model,
        tf.keras.layers.GlobalAveragePooling2D(),
        tf.keras.layers.Dense(EMB_SIZE),
        tf.keras.layers.Lambda(lambda x: tf.math.l2_normalize(x, axis=1), name='l2_embedding'),
        tf.keras.layers.Dense(len(CLASS_NAMES), activation='softmax')
    ])

    model.summary()

    model.compile(optimizer=tf.keras.optimizers.Adam(),
                  loss='sparse_categorical_crossentropy',
                  metrics=['sparse_categorical_accuracy'])

    summary_cb = tf.keras.callbacks.TensorBoard(OUTPUT_MODEL_LOGS_FOLDER, histogram_freq=1)

    model.fit(train_ds,
              epochs=EPOCHS,
              steps_per_epoch=steps_per_epoch,
              callbacks=[SaveBestValCallback()])


if __name__ == '__main__':
    main()
