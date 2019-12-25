import tensorflow as tf

from backend.net_builder import FinalLayer, NetBuilder, Arch
from backend.loss_function import combine_loss_val


def build_model(input_layer, is_training):
    builder = NetBuilder()

    net = builder.input_and_train_node(input_layer, is_training) \
        .arch_type(Arch.RES_NET50) \
        .final_layer_type(FinalLayer.G) \
        .build()

    return net


#  mode: can be one of tf.estimator.ModeKeys.{TRAIN, EVAL, PREDICT}
def model_fn(features, labels, mode, params):
    if isinstance(features, dict):  # For serving
        features = features['feature']

    is_training = (mode == tf.estimator.ModeKeys.TRAIN)

    images = features

    # -----------------------------------------------------------
    # MODEL: define the layers of the model
    with tf.variable_scope('model'):
        # Compute the l2_embeddings with the model
        l2_embeddings = build_model(images, is_training)

    with tf.variable_scope('logits'):
        logit = combine_loss_val(l2_embeddings, labels, params.num_labels, params.batch_size,
                                 params.m1, params.m2, params.m3, params.s)

    predictions = {
        'l2_embeddings': l2_embeddings,
        'classes': tf.argmax(logit, axis=1),
    }
    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)

    loss = tf.reduce_mean(
        tf.nn.sparse_softmax_cross_entropy_with_logits(
            logits=logit, labels=labels), name='inference_loss')
    accuracy = tf.metrics.accuracy(labels=labels, predictions=predictions["classes"])

    if mode == tf.estimator.ModeKeys.EVAL:
        eval_metric_ops = {"eval_accuracy": accuracy}
        return tf.estimator.EstimatorSpec(mode, loss=loss, eval_metric_ops=eval_metric_ops)

    # Summaries for training
    tf.summary.scalar('loss', loss)
    tf.summary.scalar('accuracy', accuracy[1])

    # Define training step that minimizes the loss with the Adam optimizer
    optimizer = tf.train.AdamOptimizer(params.learning_rate)
    global_step = tf.train.get_global_step()
    with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
        train_op = optimizer.minimize(loss, global_step=global_step)

    return tf.estimator.EstimatorSpec(mode, loss=loss, train_op=train_op)
