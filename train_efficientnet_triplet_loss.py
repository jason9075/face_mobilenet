import os
import pathlib

import efficientnet.tfkeras as efn
import numpy as np
import tensorflow as tf
import tensorflow_addons as tfa
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras.layers import Lambda, Dense, GlobalAveragePooling2D

import utils

tf.random.set_seed(9075)
SIZE = 112
IMG_SHAPE = (SIZE, SIZE, 3)
SHAPE = (SIZE, SIZE)
BATCH_SIZE = 128
AUTOTUNE = tf.data.experimental.AUTOTUNE
TRAIN_CLASS_NAMES = np.array([])
VALID_CLASS_NAMES = np.array([])
EPOCHS = 10000
TRAIN_DATA_PATH = 'images/train_image/'
VALID_DATA_PATH = 'images/public_face_1036_112_align/'
VER_RECORD = 'public_face_1036_112_align_verification.tfrecord'
OUTPUT_EMB_MODEL_FOLDER = 'model_out/keras_embedding'
OUTPUT_BEST_EMB_MODEL_FOLDER = 'model_out/keras_best_embedding'

MIN_IMAGES_PER_PERSON = 4
EMB_SIZE = 128

RESTORE_LAST_TRAIN = False


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


def get_label(file_path, class_name):
    parts = tf.strings.split(file_path, '/')
    one_hot = tf.cast(parts[-2] == class_name, tf.float32)
    return tf.argmax(one_hot), one_hot


def decode_img(path, aug):
    img = tf.io.read_file(path)
    img = tf.image.decode_jpeg(img, channels=3)
    if aug:
        img = tf.image.random_brightness(img, 0.2)
        img = tf.image.random_saturation(img, 0.6, 1.6)
        img = tf.image.random_contrast(img, 0.6, 1.4)
        img = tf.image.random_flip_left_right(img)
    img = tf.image.convert_image_dtype(img, tf.float32)
    img = tf.subtract(img, 0.5)
    img = tf.multiply(img, 2.0)
    return img


def process_path_train(file_path):
    label, one_hot = get_label(file_path, TRAIN_CLASS_NAMES)
    img = decode_img(file_path, True)
    return img, label


def process_path_valid(file_path):
    label, one_hot = get_label(file_path, VALID_CLASS_NAMES)
    img = decode_img(file_path, False)
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


def triplet_gen(path):
    dataset = get_dataset(path)

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
    global TRAIN_CLASS_NAMES, VALID_CLASS_NAMES

    train_list_ds = tf.data.Dataset.from_generator(
        lambda: triplet_gen(TRAIN_DATA_PATH),
        tf.string,
        (tf.TensorShape(())))

    train_data_dir = pathlib.Path(TRAIN_DATA_PATH)
    TRAIN_CLASS_NAMES = np.array(
        [item.name for item in train_data_dir.glob('*') if item.name not in [".keep", ".DS_Store"]])
    train_image_count = len(list(train_data_dir.glob('*/*.jpg')))
    steps_per_epoch = train_image_count // BATCH_SIZE

    print('total train labels: %d' % len(TRAIN_CLASS_NAMES))

    train_main_ds = train_list_ds.map(process_path_train, num_parallel_calls=AUTOTUNE)
    train_main_ds = prepare_for_training(train_main_ds)

    valid_list_ds = tf.data.Dataset.from_generator(
        lambda: triplet_gen(VALID_DATA_PATH),
        tf.string,
        (tf.TensorShape(())))

    valid_data_dir = pathlib.Path(VALID_DATA_PATH)
    VALID_CLASS_NAMES = np.array(
        [item.name for item in valid_data_dir.glob('*') if item.name not in [".keep", ".DS_Store"]])
    valid_image_count = len(list(valid_data_dir.glob('*/*.jpg')))
    valid_steps_per_epoch = valid_image_count // BATCH_SIZE

    print('total valid labels: %d' % len(VALID_CLASS_NAMES))

    valid_main_ds = valid_list_ds.map(process_path_valid, num_parallel_calls=AUTOTUNE)
    valid_main_ds = prepare_for_training(valid_main_ds)

    if RESTORE_LAST_TRAIN:
        model = tf.keras.models.load_model(OUTPUT_EMB_MODEL_FOLDER)
        model.save_weights('model_out/tmp.h5')
        base_model = efn.EfficientNetB3(input_shape=IMG_SHAPE,
                                        include_top=False, weights='imagenet')

        model = tf.keras.Sequential([
            base_model,
            GlobalAveragePooling2D(),
            Dense(EMB_SIZE, name='embedding', use_bias=False),
            Lambda(lambda x: tf.math.l2_normalize(x, axis=1), name='l2_embedding')
        ])
        model.load_weights('model_out/tmp.h5', by_name=True)
    else:
        base_model = efn.EfficientNetB2(input_shape=IMG_SHAPE,
                                        include_top=False, weights='imagenet')
        # base_model = tf.keras.applications.ResNet50V2(input_shape=IMG_SHAPE, include_top=False, weights='imagenet')

        model = tf.keras.Sequential([
            base_model,
            GlobalAveragePooling2D(),
            Dense(EMB_SIZE, name='embedding', use_bias=False),
            Lambda(lambda x: tf.math.l2_normalize(x, axis=1), name='l2_embedding')
        ])

    model.compile(optimizer=tf.keras.optimizers.Adam(0.001),
                  loss=tfa.losses.TripletSemiHardLoss(),
                  loss_weights=[1])
    model.summary()

    # iter = train_list_ds.make_one_shot_iterator()
    # next = iter.get_next()

    callbacks = [
        EarlyStopping(
            monitor='val_loss',
            patience=100,
            verbose=1,
            mode='min',
            baseline=None,
            restore_best_weights=False),
        ModelCheckpoint(
            'model_out/{epoch:02d}-{val_loss:.2f}.h5',
            monitor="val_loss",
            verbose=1,
            save_best_only=True,
            mode="min")
    ]

    model.fit(train_main_ds,
              epochs=EPOCHS,
              steps_per_epoch=steps_per_epoch,
              # callbacks=[SaveBestValCallback()],
              callbacks=callbacks,
              validation_data=valid_main_ds,
              validation_steps=valid_steps_per_epoch)


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
            sim = (sim + 1) / 2
            sim_list.append(sim[0][0])

        thresholds = np.arange(0.05, 1.0, 0.05)

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

        if self.best_acc < val_acc:
            self.model.save(OUTPUT_BEST_EMB_MODEL_FOLDER, include_optimizer=False)
            self.best_acc = val_acc
        self.model.save(OUTPUT_EMB_MODEL_FOLDER, include_optimizer=False)

        print('\n val_acc: %f, val_thr: %f, current best: %f ' % (val_acc, val_thr, self.best_acc))


if __name__ == '__main__':
    main()
