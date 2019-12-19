import os

import tensorflow as tf

from estimator.model_fn import model_fn
from estimator.input_fn import train_input_fn, test_input_fn
from estimator.utils import Params


def main():
    tf.logging.set_verbosity(tf.logging.INFO)

    json_path = os.path.join('estimator', 'params.json')
    params = Params(json_path)

    config = tf.estimator.RunConfig(tf_random_seed=9075,
                                    model_dir='model_out/resnet50',
                                    save_summary_steps=params.save_summary_steps)

    estimator = tf.estimator.Estimator(model_fn, params=params, config=config)
    estimator.train(lambda: train_input_fn('train.tfrecord', params))

    res = estimator.evaluate(lambda: test_input_fn('train.tfrecord', params))
    for key in res:
        print("{}: {}".format(key, res[key]))


if __name__ == '__main__':
    main()
