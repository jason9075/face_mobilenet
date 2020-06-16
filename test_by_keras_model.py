import glob
import os

import cv2
import numpy as np
import tensorflow as tf
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import preprocessing

import utils

SHAPE = (112, 112)
IMG_SHAPE = (112, 112, 3)
EMB_SIZE = 128
PLOT_ROC = True
LFW_PATH = 'tfrecord/lfw.bin'
RESTORE_BY_WEIGHT = False
VER_NAME = 'public_face_1036_112_align_verification.tfrecord'
VER_TYPE = 'cosine'  # euclidean, cosine


def main():
    if RESTORE_BY_WEIGHT:
        model = tf.keras.models.load_model('model_out/keras_embedding_acc901874')
        model.save_weights('model_out/keras_best/weight.h5')
        model = restore_weight('model_out/keras_best/weight.h5')
    else:
        model = tf.keras.models.load_model('model_out/keras_embedding_acc901874')

    model.summary()

    def embedding_fn(img1, img2):
        h, w, _ = img1.shape
        result1 = model.predict(np.expand_dims(img1, axis=0))
        result2 = model.predict(np.expand_dims(img2, axis=0))
        # result1 = preprocessing.normalize(result1, norm='l2')
        # result2 = preprocessing.normalize(result2, norm='l2')

        return result1, result2

    # gen_model()
    utils.test_tfrecord(VER_NAME,
                        embedding_fn,
                        SHAPE,
                        is_plot=PLOT_ROC,
                        verbose=True,
                        ver_type=VER_TYPE)
    # utils.test_lfw(LFW_PATH, embedding_fn, SHAPE, is_plot=PLOT_ROC, ver_type=VER_TYPE)


def restore_weight(path):
    base_model = tf.keras.applications.ResNet50V2(input_shape=IMG_SHAPE,
                                                  include_top=False,
                                                  weights='imagenet')

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
    model = tf.keras.models.load_model('model_out/keras_embedding_acc901874')

    img1 = pre_processing('images/train_image/0_2924088/1.jpg')
    img2 = pre_processing('images/train_image/0_2924088/2.jpg')

    result1 = model.predict(np.expand_dims(img1, axis=0))
    result2 = model.predict(np.expand_dims(img2, axis=0))

    # emb1 = preprocessing.normalize(result1, norm='l2')
    # emb2 = preprocessing.normalize(result2, norm='l2')

    if VER_TYPE == 'euclidean':
        val = np.linalg.norm(result1 - result2)
    elif VER_TYPE == 'cosine':
        val = np.dot(result1, np.transpose(result2)) / (np.sqrt(np.dot(result1, np.transpose(result1))) * np.sqrt(
            np.dot(result2, np.transpose(result2))))

        val = (val + 1) / 2
    else:
        raise RuntimeError(f'ver_type: {VER_TYPE} is not exist.')

    print(f'val: {val}')


def output_img():
    thr = 0.6
    model = tf.keras.models.load_model('model_out/keras_embedding')

    def embedding_fn(img1, img2):
        h, w, _ = img1.shape
        embedding1 = model.predict(np.expand_dims(img1, axis=0))
        embedding2 = model.predict(np.expand_dims(img2, axis=0))

        return embedding1, embedding2

    verification_path = os.path.join('tfrecord', VER_NAME)
    ver_dataset = utils.get_ver_data(verification_path, SHAPE)

    first_list, second_list, true_same, first_name_list, second_name_list = \
        ver_dataset[0], ver_dataset[1], np.array(ver_dataset[2]), ver_dataset[3], ver_dataset[4]

    for idx, (first, second, gt, fn, sn) in enumerate(
            zip(first_list, second_list, true_same, first_name_list, second_name_list)):
        result1, result2 = embedding_fn(first, second)

        val = np.dot(result1, np.transpose(result2)) / (np.sqrt(np.dot(result1, np.transpose(result1))) * np.sqrt(
            np.dot(result2, np.transpose(result2))))[0][0]

        val = (val + 1.0) / 2

        val_percent = int(val * 100)
        first = first / 0.0078125 + 127.5
        second = second / 0.0078125 + 127.5
        fn = fn.decode("utf-8").replace('/', '-')
        sn = sn.decode("utf-8").replace('/', '-')
        if thr < val:  # predict result is same
            if gt:
                res_type = 'tp'
            else:
                res_type = 'fp'
        else:  # predict result is not same
            if gt:
                res_type = 'fn'
            else:
                res_type = 'tn'
        cv2.imwrite(f'images/ver/{res_type}_{idx}_{val_percent}_{fn}', first)
        cv2.imwrite(f'images/ver/{res_type}_{idx}_{val_percent}_{sn}', second)

        if idx == 1000:
            break


def tsne():
    model = tf.keras.models.load_model('model_out/keras_embedding_acc901874')

    labels = []
    vectors = []
    for img_path in glob.glob('images/train_image/*/*.jpg'):
        labels.append(img_path.split('/')[-2])
        img = pre_processing(img_path)
        vector = model.predict(np.expand_dims(img, axis=0))
        vectors.append(vector[0])

    embedded = TSNE(n_components=2,
                    verbose=1,
                    perplexity=8,
                    n_iter=5000).fit_transform(vectors)

    plt.figure(figsize=(16, 10))
    unique = np.unique(labels)
    markers = ['o', '^', '1', 'p', 'P', 'X', 'D']
    plt.title('tsne')
    colors = [
        plt.cm.jet(i / float(len(unique) - 1)) for i in range(len(unique))
    ]
    colors = [np.asarray(c).reshape((1, -1)) for c in colors]
    for i, u in enumerate(unique):
        x0 = [embedded[j][0] for j in range(len(labels)) if labels[j] == u]
        x1 = [embedded[j][1] for j in range(len(labels)) if labels[j] == u]
        plt.scatter(
            x0,
            x1,
            c=colors[i],
            s=80,
            label=str(u),
            marker=markers[i % len(markers)])

    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.show()


if __name__ == '__main__':
    # main()
    # single_test()
    # output_img()
    tsne()
