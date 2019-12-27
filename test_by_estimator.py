import os

import tensorflow as tf
from tensorflow.contrib import predictor

import utils
from estimator.input_fn import serving_input_receiver_fn
from estimator.model_fn import model_fn
from estimator.utils import Params

SHAPE = (224, 224)
PLOT_ROC = True
LFW_PATH = 'tfrecord/lfw.bin'
SAVED_MODEL_PATH = 'model_out/saved_model/1577169053'


def main():
    # if you want to test on cpu environment. please check below link.
    # https://github.com/tensorflow/tensorflow/issues/17149
    predict_fn = predictor.from_saved_model(SAVED_MODEL_PATH,
                                            config=tf.ConfigProto(allow_soft_placement=True))

    # estimator method
    def embedding_fn(img1, img2):
        result1 = predict_fn({'input_image': [img1]})['l2_embeddings']
        result2 = predict_fn({'input_image': [img2]})['l2_embeddings']
        return result1, result2

    # gen_model()
    # utils.test_tfrecord('verification.tfrecord', embedding_fn, SHAPE, is_plot=PLOT_ROC)
    utils.test_lfw(LFW_PATH, embedding_fn, SHAPE, is_plot=PLOT_ROC)


def gen_model():
    json_path = os.path.join('estimator', 'params.json')
    params = Params(json_path)
    estimator = tf.estimator.Estimator(model_fn, model_dir='model_out/resnet50/', params=params)
    estimator.export_saved_model('model_out/saved_model/', serving_input_receiver_fn)


if __name__ == '__main__':
    main()
