from backend.layers import *
import tensorflow as tf


def combine_loss_val(embedding, gt_labels, num_labels, batch_size, m1, m2, m3, s):
    with tf.variable_scope('combine_loss'):
        ordinal = tf.range(batch_size)
        ordinal_y = tf.stack([ordinal, gt_labels], axis=1)

        embedding_size = embedding.get_shape().as_list()[-1]
        shape = (embedding_size, num_labels)

        weights = tf.get_variable(name='w_embedding', shape=shape,
                                  initializer=WEIGHT_INIT, dtype=D_TYPE)
        weights = tf.nn.l2_normalize(weights, axis=0)
        embedding = tf.nn.l2_normalize(embedding, axis=1)

        cos_t = tf.matmul(embedding, weights)
        if m1 == 1.0 and m2 == 0.0 and m3 == 0:
            return cos_t * s  # pure softmax

        cos_t = cos_t * s
        zy = tf.gather_nd(cos_t, ordinal_y)
        if m1 == 1.0 and m2 == 0.0:  # cosine face only
            s_m3 = s * m3
            new_zy = zy - s_m3
        else:
            cos_value = zy / s
            t = tf.acos(cos_value)
            t = t * m1  # sphere
            t = t + m2  # arc
            new_cos_value = tf.cos(t)
            new_cos_value = new_cos_value - m3  # cos
            new_zy = new_cos_value * s
        diff = tf.subtract(new_zy, zy)
        body = tf.scatter_nd(ordinal_y, diff, cos_t.get_shape())

        updated_logits = tf.add(cos_t, body)

    return updated_logits


def loss_val():
    # n=10
    emb_size = 2
    num_class = 3
    gt_labels = [0, 1, 1, 2, 2, 0, 1, 0, 2, 2]
    emb = [[0.1, 0.7], [0.2, 0.3], [0.6, 0.4], [0.2, 0.5], [0.7, 0.9],
           [0.4, 0.8], [0.1, 0.3], [0.3, 0.3], [0.6, 0.5], [0.5, 0.3]]

    weight = tf.get_variable(name='weight', shape=(emb_size, num_class),
                             initializer=WEIGHT_INIT, dtype=D_TYPE)

    paras = (1.0, 0.2, 0.3, 64)  # m1,m2,m3,s

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        print(f'weight:\n{sess.run(weight)}\n\n')

        fc7_my = my_impl(gt_labels, emb, weight, (1.0, 0.0, 0.0, 64))
        print(f'origin:\n{sess.run(fc7_my)}\n')

        fc7_mxnet = mxnet_function(gt_labels, num_class, emb, weight, paras)
        print(f'mx:\n{sess.run(fc7_mxnet)}\n')
        fc7_my = my_impl(gt_labels, emb, weight, paras)
        print(f'my:\n{sess.run(fc7_my)}\n')


def mxnet_function(gt_labels, num_class, embedding, weights, paras):
    m1, m2, m3, s = paras
    ordinal = tf.range(len(gt_labels))
    ordinal_y = tf.stack([ordinal, gt_labels], axis=1)

    weights = tf.nn.l2_normalize(weights, axis=0)
    nembedding = tf.nn.l2_normalize(embedding, axis=1)
    nembedding = nembedding * s

    fc7 = tf.matmul(nembedding, weights)

    if m1 != 1.0 or m2 != 0.0 or m3 != 0.0:
        if m1 == 1.0 and m2 == 0.0:
            s_m = s * m3
            gt_one_hot = tf.one_hot(gt_labels, num_class)
            fc7 = fc7 - gt_one_hot * s_m
        else:
            zy = tf.gather_nd(fc7, ordinal_y)
            cos_t = zy / s
            t = tf.acos(cos_t)
            if m1 != 1.0:
                t = t * m1
            if m2 > 0.0:
                t = t + m2
            body = tf.cos(t)
            if m3 > 0.0:
                body = body - m3
            new_zy = body * s
            diff = new_zy - zy
            diff = tf.expand_dims(diff, 1)
            gt_one_hot = tf.one_hot(gt_labels, num_class)
            body = tf.math.multiply(gt_one_hot, diff)
            fc7 = fc7 + body

    return fc7


def my_impl(gt_labels, embedding, weights, paras):
    m1, m2, m3, s = paras
    ordinal = tf.range(len(gt_labels))
    ordinal_y = tf.stack([ordinal, gt_labels], axis=1)

    weights = tf.nn.l2_normalize(weights, axis=0)
    embedding = tf.nn.l2_normalize(embedding, axis=1)

    cos_t = tf.matmul(embedding, weights)
    if m1 == 1.0 and m2 == 0.0 and m3 == 0:
        return cos_t * s  # pure softmax

    cos_t = cos_t * s
    zy = tf.gather_nd(cos_t, ordinal_y)
    if m1 == 1.0 and m2 == 0.0:  # cosine face only
        s_m3 = s * m3
        new_zy = zy - s_m3
    else:
        cos_value = zy / s
        t = tf.acos(cos_value)
        t = t * m1  # sphere
        t = t + m2  # arc
        new_cos_value = tf.cos(t)
        new_cos_value = new_cos_value - m3  # cos
        new_zy = new_cos_value * s
    diff = tf.subtract(new_zy, zy)
    body = tf.scatter_nd(ordinal_y, diff, cos_t.get_shape())

    updated_logits = tf.add(cos_t, body)
    return updated_logits


if __name__ == '__main__':
    loss_val()
