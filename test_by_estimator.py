import os
from pathlib import Path

import cv2
import tensorflow as tf
from tensorflow.contrib import predictor
import numpy as np

import utils
from estimator.input_fn import serving_input_receiver_fn
from estimator.model_fn import model_fn
from estimator.utils import Params


def main():
    # gen_model()
    test_model()


def test_model():
    export_dir = 'model_out/saved_model'
    subdirs = [x for x in Path(export_dir).iterdir()
               if x.is_dir() and 'temp' not in str(x)]
    latest = str(sorted(subdirs)[-1])
    latest = 'model_out/saved_model/1577284216'
    # if you want to test on cpu environment. please check below link.
    # https://github.com/tensorflow/tensorflow/issues/17149
    predict_fn = predictor.from_saved_model(latest,
                                            config=tf.ConfigProto(allow_soft_placement=True))

    # img1 = cv2.imread('images/star224/Arak Amornsuppasiri/000001_0.jpg')
    # img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
    # img2 = cv2.imread('images/star224/Arak Amornsuppasiri/000004_0.jpg')
    # img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)
    # result1 = predict_fn({'input_image': [img1]})['l2_embeddings']
    # result2 = predict_fn({'input_image': [img2]})['l2_embeddings']
    # print(result1)
    # print(result2)
    # dist = np.linalg.norm(result1 - result2)
    # print(dist)

    SHAPE = (224, 224)
    verification_path = os.path.join('tfrecord', 'verification.tfrecord')
    ver_dataset = utils.get_ver_data(verification_path, SHAPE)
    val_acc, val_thr, val_prec, val_rec = utils.ver_est_test(ver_dataset, predict_fn)
    print('test accuracy is: %.3f, thr: %.2f, val_prec: %.3f, val_rec: %.3f.' %
          (val_acc, val_thr, val_prec, val_rec))


def gen_model():
    json_path = os.path.join('estimator', 'params.json')
    params = Params(json_path)
    estimator = tf.estimator.Estimator(model_fn, model_dir='model_out/resnet50/', params=params)
    estimator.export_saved_model('model_out/saved_model/', serving_input_receiver_fn)


if __name__ == '__main__':
    main()
