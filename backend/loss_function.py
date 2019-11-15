from backend.layers import *
import tensorflow as tf


def combine_loss_val(embedding, gt_labels, num_labels, batch_size, m1, m2, m3,
                     s):
    with tf.variable_scope('combine_loss'):
        ordinal = tf.range(batch_size, dtype=tf.int64)
        ordinal_y = tf.stack([ordinal, gt_labels], axis=1)

        embedding_size = embedding.get_shape().as_list()[-1]
        shape = (embedding_size, num_labels)

        weights = tf.get_variable(
            name='w_embedding',
            shape=shape,
            initializer=WEIGHT_INIT,
            dtype=D_TYPE)
        weights = tf.nn.l2_normalize(weights, axis=0)
        embedding = tf.nn.l2_normalize(embedding, axis=1)

        cos_t = tf.matmul(
            embedding, weights, name='embedding_dense')  # fully connect dense
        if m1 == 1.0 and m2 == 0.0 and m3 == 0:
            return tf.scalar_mul(s, cos_t)  # pure softmax

        cos_t = tf.scalar_mul(s, cos_t)
        zy = tf.gather_nd(cos_t, ordinal_y)
        if m1 == 1.0 and m2 == 0.0:  # cosine face only
            s_m3 = tf.scalar_mul(s, m3)
            new_zy = zy - s_m3
        else:
            cos_value = tf.divide(zy, s)
            t = tf.acos(cos_value)
            t = t * m1  # sphere
            t = t + m2  # arc
            new_cos_value = tf.cos(t)
            new_cos_value = tf.subtract(new_cos_value, m3)  # cos
            new_zy = tf.scalar_mul(s, new_cos_value)
        diff = tf.subtract(new_zy, zy)
        body = tf.scatter_nd(ordinal_y, diff,
                             (batch_size, cos_t.get_shape()[1]))

        updated_logits = tf.add(cos_t, body, name='combine_loss_output')

    return updated_logits


def pure_softmax(embedding, num_labels):
    with tf.variable_scope('pure_softmax'):
        embedding_size = embedding.get_shape().as_list()[-1]
        shape = (embedding_size, num_labels)

        weights = tf.get_variable(
            name='class_weight',
            shape=shape,
            initializer=WEIGHT_INIT,
            dtype=D_TYPE)

    return tf.matmul(
        embedding, weights, name='embedding_dense')  # fully connect dense


def triplet_loss(anchor, positive, negative, alpha):
    with tf.variable_scope('triplet_loss'):
        pos_dist = tf.reduce_sum(tf.square(tf.subtract(anchor, positive)), 1)
        neg_dist = tf.reduce_sum(tf.square(tf.subtract(anchor, negative)), 1)

        basic_loss = tf.add(tf.subtract(pos_dist, neg_dist), alpha)
        loss = tf.reduce_sum(tf.maximum(basic_loss, 0.0), 0)

    return loss


def loss_val():
    # n=10
    # emb_size = 2
    num_class = 3
    gt_labels = [0, 1, 1, 2, 2, 0, 1, 0, 2, 2]
    emb = [[0.1, 0.7], [0.2, 0.3], [0.6, 0.4], [0.2, 0.5], [0.7, 0.9],
           [0.4, 0.8], [0.1, 0.3], [0.3, 0.3], [0.6, 0.5], [0.5, 0.3]]

    weight = [[0.0072331, -0.01129611, -0.00556139],
              [0.00785534, 0.02351345, 0.01411054]]

    weight_mx = np.transpose(weight)

    paras = (1.0, 0.2, 0.3, 64)  # m1,m2,m3,s

    mxnet_function(gt_labels, num_class, emb, weight_mx, paras)

    my_impl(gt_labels, emb, weight, paras)


def mxnet_function(gt_labels, num_class, embedding, weights, paras):
    import mxnet as mx

    m1, m2, m3, s = paras
    gt_label = mx.sym.Variable('gt_label')
    _weight = mx.symbol.L2Normalization(
        mx.sym.Variable('weight'), mode='instance')
    nembedding = mx.symbol.L2Normalization(
        mx.sym.Variable('embedding'), mode='instance', name='fc1n') * s

    fc7 = mx.sym.FullyConnected(
        data=nembedding,
        weight=_weight,
        no_bias=True,
        num_hidden=num_class,
        name='fc7')
    if m1 != 1.0 or m2 != 0.0 or m3 != 0.0:
        if m1 == 1.0 and m2 == 0.0:
            s_m = s * m3
            gt_one_hot = mx.sym.one_hot(
                gt_label, depth=num_class, on_value=s_m, off_value=0.0)
            fc7 = fc7 - gt_one_hot
        else:
            zy = mx.sym.pick(fc7, gt_label, axis=1)
            cos_t = zy / s
            t = mx.sym.arccos(cos_t)
            if m1 != 1.0:
                t = t * m1
            if m2 > 0.0:
                t = t + m2
            body = mx.sym.cos(t)
            if m3 > 0.0:
                body = body - m3
            new_zy = body * s
            diff = new_zy - zy
            diff = mx.sym.expand_dims(diff, 1)
            gt_one_hot = mx.sym.one_hot(
                gt_label, depth=num_class, on_value=1.0, off_value=0.0)
            body = mx.sym.broadcast_mul(gt_one_hot, diff)
            fc7 = fc7 + body

    e = fc7.bind(
        mx.cpu(), {
            'weight': mx.nd.array(weights),
            'embedding': mx.nd.array(embedding),
            'gt_label': mx.nd.array(gt_labels)
        })
    print('mx\n', e.forward())


def my_impl(gt_labels, embedding, weights, paras):
    with tf.Session() as sess:
        m1, m2, m3, s = paras
        ordinal = tf.range(len(gt_labels))
        ordinal_y = tf.stack([ordinal, gt_labels], axis=1)

        weights = tf.nn.l2_normalize(weights, axis=0)
        embedding = tf.nn.l2_normalize(embedding, axis=1)

        cos_t = tf.matmul(embedding, weights)
        if m1 == 1.0 and m2 == 0.0 and m3 == 0:
            updated_logits = cos_t * s
            print('my:\n', sess.run(updated_logits))
            return

        cos_t = tf.scalar_mul(s, cos_t)
        zy = tf.gather_nd(cos_t, ordinal_y)
        if m1 == 1.0 and m2 == 0.0:  # cosine face only
            s_m3 = tf.scalar_mul(s, m3)
            new_zy = zy - s_m3
        else:
            cos_value = tf.divide(zy, s)
            t = tf.acos(cos_value)
            t = t * m1  # sphere
            t = t + m2  # arc
            new_cos_value = tf.cos(t)
            new_cos_value = tf.subtract(new_cos_value, m3)  # cos
            new_zy = tf.scalar_mul(s, new_cos_value)
        diff = tf.subtract(new_zy, zy)
        body = tf.scatter_nd(ordinal_y, diff, cos_t.get_shape())

        updated_logits = tf.add(cos_t, body)
        print('my:\n', sess.run(updated_logits))


if __name__ == '__main__':
    loss_val()
