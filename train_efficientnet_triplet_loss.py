import os
import pathlib

import efficientnet.tfkeras as efn
import numpy as np
import tensorflow as tf
import tensorflow_addons as tfa
from tensorflow_core.python.keras.layers import Dense, Lambda

import utils

tf.random.set_seed(9075)
SIZE = 112
IMG_SHAPE = (SIZE, SIZE, 3)
SHAPE = (SIZE, SIZE)
BATCH_SIZE = 128
AUTOTUNE = tf.data.experimental.AUTOTUNE
CLASS_NAMES = np.array([])
EPOCHS = 10000
TRAIN_DATA_PATH = 'images/train_image/'
VER_RECORD = 'public_face_1036_112_train_align.tfrecord'
OUTPUT_EMB_MODEL_FOLDER = 'model_out/keras_embedding'

MIN_IMAGES_PER_PERSON = 4
EMB_SIZE = 128


class ImageClass:
    def __init__(self, name, image_paths):
        self.name = name
        self.image_paths = image_paths

    def __str__(self):
        return self.name + ', ' + str(len(self.image_paths)) + ' images'

    def __len__(self):
        return len(self.image_paths)


def get_image_paths(facedir):
    image_paths = []
    if os.path.isdir(facedir):
        images = os.listdir(facedir)
        images = [image for image in images if image.endswith('.jpg')]
        image_paths = [os.path.join(facedir, img) for img in images]
    return image_paths


def get_label(file_path):
    parts = tf.strings.split(file_path, '/')
    one_hot = tf.cast(parts[-2] == CLASS_NAMES, tf.float32)
    return tf.argmax(one_hot), one_hot


def decode_img(path):
    img = tf.io.read_file(path)
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
    label, one_hot = get_label(file_path)
    img = decode_img(file_path)
    return img, label


def prepare_for_training(ds, cache=False, shuffle_buffer_size=2000):
    if cache:
        if isinstance(cache, str):
            ds = ds.cache(cache)
        else:
            ds = ds.cache()

    ds = ds.batch(BATCH_SIZE)
    ds = ds.prefetch(buffer_size=AUTOTUNE)

    return ds


def get_dataset(path):
    dataset = []
    path_exp = os.path.expanduser(path)
    classes = [path for path in os.listdir(path_exp) \
               if os.path.isdir(os.path.join(path_exp, path))]
    classes.sort()
    nrof_classes = len(classes)
    for i in range(nrof_classes):
        class_name = classes[i]
        facedir = os.path.join(path_exp, class_name)
        image_paths = get_image_paths(facedir)
        dataset.append(ImageClass(class_name, image_paths))

    dataset = [data for data in dataset if MIN_IMAGES_PER_PERSON <= len(data)]
    return dataset


def triplet_gen():
    dataset = get_dataset(TRAIN_DATA_PATH)

    while True:
        np.random.shuffle(dataset)
        for ic_idx in range(0, len(dataset)):
            ic = dataset[ic_idx]
            path_list = ic.image_paths
            np.random.shuffle(path_list)
            for path_idx in range(0, MIN_IMAGES_PER_PERSON):
                img_path = path_list[path_idx]
                yield img_path


def main():
    global CLASS_NAMES

    train_list_ds = tf.data.Dataset.from_generator(
        triplet_gen,
        tf.string,
        (tf.TensorShape(())))

    train_data_dir = pathlib.Path(TRAIN_DATA_PATH)
    CLASS_NAMES = np.array([item.name for item in train_data_dir.glob('*') if item.name not in [".keep", ".DS_Store"]])
    train_image_count = len(list(train_data_dir.glob('*/*.jpg')))
    steps_per_epoch = np.ceil(train_image_count / BATCH_SIZE)

    print('total labels: %d' % len(CLASS_NAMES))

    train_main_ds = train_list_ds.map(process_path, num_parallel_calls=AUTOTUNE)
    train_main_ds = prepare_for_training(train_main_ds)

    base_model = efn.EfficientNetB3(input_shape=IMG_SHAPE,
                                    include_top=False, weights='imagenet')

    model = tf.keras.Sequential([
        base_model,
        tf.keras.layers.GlobalAveragePooling2D(),
        Dense(EMB_SIZE, name='embedding', use_bias=False),
        Lambda(lambda x: tf.math.l2_normalize(x, axis=1), name='l2_embedding')
    ])

    model.compile(optimizer=tf.keras.optimizers.Adam(0.001),
                  loss=tfa.losses.TripletSemiHardLoss(),
                  loss_weights=[1])
    model.summary()

    # iter = train_list_ds.make_one_shot_iterator()
    # next = iter.get_next()

    model.fit(train_main_ds,
              epochs=EPOCHS,
              steps_per_epoch=steps_per_epoch,
              callbacks=[SaveBestValCallback()])


class SaveBestValCallback(tf.keras.callbacks.Callback):

    def __init__(self):
        super().__init__()
        self.embedding_model = None
        self.best_acc = 0

    def set_model(self, model):
        self.model = model
        embedding = model.get_layer(name='l2_embedding')
        self.embedding_model = tf.keras.models.Model(inputs=model.input,
                                                     outputs=embedding.output)

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

            sim = np.dot(result1, np.transpose(result2)) / (np.sqrt(np.dot(result1, np.transpose(result1))) * np.sqrt(
                np.dot(result2, np.transpose(result2))))
            sim_list.append(sim[0][0])

        thresholds = np.arange(0.1, 1.0, 0.05)

        accs = []
        for threshold in thresholds:
            pred_same = np.greater(sim_list, threshold)
            tp = np.sum(np.logical_and(pred_same, true_same))
            tn = np.sum(np.logical_and(np.logical_not(pred_same), np.logical_not(true_same)))
            acc = float(tp + tn) / total
            accs.append(acc)
        best_index = int(np.argmax(accs))

        val_acc = accs[best_index]
        val_thr = thresholds[best_index]

        print('\n val_acc: %f, val_thr: %f, current best: %f ' % (val_acc, val_thr, self.best_acc))
        if self.best_acc < val_acc:
            self.embedding_model.save(OUTPUT_EMB_MODEL_FOLDER)
            self.best_acc = val_acc


if __name__ == '__main__':
    main()
