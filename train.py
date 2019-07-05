import glob
import os
import time

import cv2
import numpy as np
import tensorflow as tf
from tensorflow.core.protobuf import config_pb2

import utils
from backend.loss_function import combine_loss_val
from backend.mobilenet_v2 import MobileNetV2

MODEL_OUT_PATH = os.path.join('model_out')


def purge():
    for f in glob.glob(os.path.join('events/events*')):
        os.remove(f)


def main():
    num_classes = 85742
    batch_size = 32
    buffer_size = 30000
    epoch = 10000
    lr = 0.001
    saver_max_keep = 10
    momentum = 0.99
    show_info_interval = 100
    summary_interval = 200
    ckpt_interval = 1000
    validate_interval = 2000
    input_size = (112, 112)

    purge()

    record_path = os.path.join('tfrecord', 'train.tfrecord')
    data_set = tf.data.TFRecordDataset(record_path)
    data_set = data_set.map(utils.parse_function)
    data_set = data_set.shuffle(buffer_size=buffer_size)
    data_set = data_set.batch(batch_size)
    iterator = data_set.make_initializable_iterator()
    next_element = iterator.get_next()

    verification_path = os.path.join('tfrecord', 'verification.tfrecord')
    ver_dataset = utils.get_ver_data(verification_path)

    with tf.Session() as sess:

        global_step = tf.Variable(name='global_step', initial_value=0, trainable=False)
        inc_op = tf.assign_add(global_step, 1, name='increment_global_step')
        input_layer = tf.placeholder(name='input_images', shape=[None, input_size[0], input_size[1], 3],
                                     dtype=tf.float32)
        labels = tf.placeholder(name='img_labels', shape=[None, ], dtype=tf.int64)
        trainable = tf.placeholder(name='trainable_bn', dtype=tf.bool)

        net = MobileNetV2(input_layer, trainable)

        logit = combine_loss_val(embedding=net.embedding, gt_labels=labels, num_labels=num_classes,
                                 batch_size=batch_size, m1=1, m2=0, m3=0, s=64)
        inference_loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logit, labels=labels))

        wd_loss = tf.constant(0, name='wd', dtype=tf.float32)
        # for weights in tl.layers.get_variables_with_name('W_conv2d', True, True):
        #     wd_loss += tf.contrib.layers.l2_regularizer(args.weight_deacy)(weights)
        # for W in tl.layers.get_variables_with_name('resnet_v1_50/E_DenseLayer/W', True, True):
        #     wd_loss += tf.contrib.layers.l2_regularizer(args.weight_deacy)(W)
        # for weights in tl.layers.get_variables_with_name('embedding_weights', True, True):
        #     wd_loss += tf.contrib.layers.l2_regularizer(args.weight_deacy)(weights)
        # for gamma in tl.layers.get_variables_with_name('gamma', True, True):
        #     wd_loss += tf.contrib.layers.l2_regularizer(args.weight_deacy)(gamma)
        # for beta in tl.layers.get_variables_with_name('beta', True, True):
        #     wd_loss += tf.contrib.layers.l2_regularizer(args.weight_deacy)(beta)
        # for alphas in tl.layers.get_variables_with_name('alphas', True, True):
        #     wd_loss += tf.contrib.layers.l2_regularizer(args.weight_deacy)(alphas)
        # for bias in tl.layers.get_variables_with_name('resnet_v1_50/E_DenseLayer/b', True, True):
        #     wd_loss += tf.contrib.layers.l2_regularizer(args.weight_deacy)(bias)

        total_loss = inference_loss + wd_loss

        opt = tf.train.MomentumOptimizer(learning_rate=lr, momentum=momentum)
        grads = opt.compute_gradients(total_loss)

        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            train_op = opt.apply_gradients(grads, global_step=global_step)

        pred = tf.nn.softmax(logit)
        acc = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(pred, axis=1), labels), dtype=tf.float32))

        summary = tf.summary.FileWriter('events/', sess.graph)
        summaries = []

        for grad, var in grads:
            if grad is not None:
                summaries.append(tf.summary.histogram(var.op.name + '/gradients', grad))

        for var in tf.trainable_variables():
            summaries.append(tf.summary.histogram(var.op.name, var))

        summaries.append(tf.summary.scalar('inference_loss', inference_loss))
        summaries.append(tf.summary.scalar('wd_loss', wd_loss))
        summaries.append(tf.summary.scalar('total_loss', total_loss))
        summaries.append(tf.summary.scalar('leraning_rate', lr))
        summary_op = tf.summary.merge(summaries)
        saver = tf.train.Saver(max_to_keep=saver_max_keep)

        sess.run(tf.global_variables_initializer())
        count = 0
        best_accuracy = 0
        for i in range(epoch):
            sess.run(iterator.initializer)
            while True:
                try:
                    images_train, labels_train = sess.run(next_element)
                    feed_dict = {input_layer: images_train, labels: labels_train, trainable: True}
                    start = time.time()
                    _, total_loss_val, inference_loss_val, wd_loss_val, _, acc_val = \
                        sess.run([train_op, total_loss, inference_loss, wd_loss, inc_op, acc],
                                 feed_dict=feed_dict,
                                 options=config_pb2.RunOptions(report_tensor_allocations_upon_oom=True))
                    end = time.time()
                    pre_sec = batch_size / (end - start)
                    # print training information
                    if count > 0 and count % show_info_interval == 0:
                        print('epoch %d, total_step %d, total loss is %.2f , inference loss is %.2f, weight deacy '
                              'loss is %.2f, training accuracy is %.6f, time %.3f samples/sec' %
                              (i, count, total_loss_val, inference_loss_val, wd_loss_val, acc_val, pre_sec))
                    count += 1
                    # save summary
                    if count > 0 and count % summary_interval == 0:
                        feed_summary_dict = {input_layer: images_train, labels: labels_train, trainable: False}
                        summary_op_val = sess.run(summary_op, feed_dict=feed_summary_dict)
                        summary.add_summary(summary_op_val, count)

                    # save ckpt files
                    if count > 0 and count % ckpt_interval == 0:
                        print('epoch: %d,count: %d, saving ckpt.' % (i, count))
                        filename = 'InsightFace_iter_{:d}'.format(count) + '.ckpt'
                        filename = os.path.join(MODEL_OUT_PATH, filename)
                        saver.save(sess, filename)

                    # validate
                    if count > 0 and count % validate_interval == 0:
                        feed_dict_test = {trainable: False}
                        val_acc, val_thr = utils.ver_test(data_set=ver_dataset,
                                                          sess=sess,
                                                          embedding_tensor=net.embedding,
                                                          feed_dict=feed_dict_test,
                                                          input_placeholder=input_layer)
                        print('test accuracy is: ', str(val_acc), ', thr: ', str(val_thr))
                        if 0.7 < val_acc and best_accuracy < val_acc:
                            print('best accuracy is %.5f' % val_acc)
                            best_accuracy = val_acc
                            filename = 'InsightFace_iter_best_{:d}'.format(count) + '.ckpt'
                            filename = os.path.join(MODEL_OUT_PATH, filename)
                            saver.save(sess, filename)
                except tf.errors.OutOfRangeError:
                    print("End of epoch %d" % i)
                    break
                except KeyboardInterrupt:
                    print('KeyboardInterrupt, saving ckpt.')
                    filename = 'InsightFace_iter_{:d}'.format(count) + '.ckpt'
                    filename = os.path.join(MODEL_OUT_PATH, filename)
                    saver.save(sess, filename)
                    exit(0)


def test():
    with tf.Session() as sess:
        saver = tf.train.import_meta_graph('InsightFace_iter_84350.ckpt.meta', clear_devices=True)
        saver.restore(sess, "InsightFace_iter_84350.ckpt")

        image1 = cv2.imread('images/image_db/andy/26bb7b_1.jpg')
        image2 = cv2.imread('images/image_db/rivon/gen_9f2816_5.jpg')

        image1 = processing(image1)
        image2 = processing(image2)

        input_tensor = tf.get_default_graph().get_tensor_by_name("input_images:0")
        trainable = tf.get_default_graph().get_tensor_by_name("trainable_bn:0")
        embedding_tensor = tf.get_default_graph().get_tensor_by_name("gdc/embedding/Identity:0")

        feed_dict = {input_tensor: np.expand_dims(image1, 0), trainable: False}
        vector1 = sess.run(embedding_tensor, feed_dict=feed_dict)

        feed_dict = {input_tensor: np.expand_dims(image2, 0), trainable: False}
        vector2 = sess.run(embedding_tensor, feed_dict=feed_dict)

        print(vector1)
        print(vector2)

        # print(f'dist: {np.linalg.norm(vector1 - vector2)}')


def processing(img):
    img = cv2.resize(img, (112, 112))
    # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = np.array(img) - 127.5
    img *= 0.0078125
    return img


if __name__ == '__main__':
    main()
    # test()
