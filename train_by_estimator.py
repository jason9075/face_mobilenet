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
                                    save_summary_steps=params.save_summary_steps,
                                    save_checkpoints_steps=params.save_checkpoints_steps,
                                    log_step_count_steps=params.log_steps)

    estimator = tf.estimator.Estimator(model_fn, params=params, config=config)
    early_stopping = tf.contrib.estimator.stop_if_no_decrease_hook(
        estimator,
        metric_name='loss',
        max_steps_without_decrease=1000,
        min_steps=10000)

    train_spec = tf.estimator.TrainSpec(input_fn=lambda: train_input_fn('train_1036.tfrecord', params),
                                        hooks=[early_stopping])
    eval_spec = tf.estimator.EvalSpec(input_fn=lambda: test_input_fn('valid_1036.tfrecord', params),
                                      steps=None, throttle_secs=60, start_delay_secs=0)
    tf.estimator.train_and_evaluate(estimator, train_spec, eval_spec)


if __name__ == '__main__':
    main()
