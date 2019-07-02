import glob
import os
import time

import tensorflow as tf
from tensorflow.core.protobuf import config_pb2

import utils
from backend.loss_function import combine_loss_val
from backend.mobilenet_v2 import MobileNetV2


def purge():
    for f in glob.glob(os.path.join('events/events*')):
        os.remove(f)


def main():
    num_classes = 19
    batch_size = 20
    buffer_size = 10000
    epoch = 100
    lr = 0.001
    saver_max_keep = 5
    momentum = 0.99
    show_info_interval = 20
    summary_interval = 300
    input_size = (224, 224)

    purge()

    record_path = os.path.join('tfrecord', 'train.tfrecord')
    data_set = tf.data.TFRecordDataset(record_path)
    data_set = data_set.map(utils.parse_function)
    data_set = data_set.shuffle(buffer_size=buffer_size)
    data_set = data_set.batch(batch_size)
    iterator = data_set.make_initializable_iterator()
    next_element = iterator.get_next()

    with tf.Session() as sess:

        global_step = tf.Variable(name='global_step', initial_value=0, trainable=False)
        inc_op = tf.assign_add(global_step, 1, name='increment_global_step')
        input_layer = tf.placeholder(name='input_images', shape=[None, input_size[0], input_size[1], 3], dtype=tf.float32)
        labels = tf.placeholder(name='img_labels', shape=[None, ], dtype=tf.int64)
        trainable = tf.placeholder(name='trainable_bn', dtype=tf.bool)

        net = MobileNetV2(input_layer)

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
        # 3.11.2 add trainabel variable gradients
        for var in tf.trainable_variables():
            summaries.append(tf.summary.histogram(var.op.name, var))
        # 3.11.3 add loss summary
        summaries.append(tf.summary.scalar('inference_loss', inference_loss))
        summaries.append(tf.summary.scalar('wd_loss', wd_loss))
        summaries.append(tf.summary.scalar('total_loss', total_loss))
        # 3.11.4 add learning rate
        summaries.append(tf.summary.scalar('leraning_rate', lr))
        summary_op = tf.summary.merge(summaries)
        # 3.12 saver
        saver = tf.train.Saver(max_to_keep=saver_max_keep)
        # 3.13 init all variables
        sess.run(tf.global_variables_initializer())

        # 4 begin iteration
        count = 0
        total_accuracy = {}

        for i in range(epoch):
            sess.run(iterator.initializer)
            while True:
                try:
                    images_train, labels_train = sess.run(next_element)
                    print(labels_train)
                    feed_dict = {input_layer: images_train, labels: labels_train}
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
                        feed_dict = {images_train: images_train, labels: labels_train}
                        summary_op_val = sess.run(summary_op, feed_dict=feed_dict)
                        summary.add_summary(summary_op_val, count)

                    # # save ckpt files
                    # if count > 0 and count % args.ckpt_interval == 0:
                    #     filename = 'InsightFace_iter_{:d}'.format(count) + '.ckpt'
                    #     filename = os.path.join(args.ckpt_path, filename)
                    #     saver.save(sess, filename)
                    #
                    # # validate
                    # if count > 0 and count % args.validate_interval == 0:
                    #     feed_dict_test = {dropout_rate: 1.0}
                    #     feed_dict_test.update(tl.utils.dict_to_one(net.all_drop))
                    #     results = ver_test(ver_list=ver_list, ver_name_list=ver_name_list, nbatch=count, sess=sess,
                    #                        embedding_tensor=embedding_tensor, batch_size=batch_size,
                    #                        feed_dict=feed_dict_test,
                    #                        input_placeholder=images)
                    #     print('test accuracy is: ', str(results[0]))
                    #     total_accuracy[str(count)] = results[0]
                    #     if max(results) > 0.996:
                    #         print('best accuracy is %.5f' % max(results))
                    #         filename = 'InsightFace_iter_best_{:d}'.format(count) + '.ckpt'
                    #         filename = os.path.join(args.ckpt_path, filename)
                    #         saver.save(sess, filename)
                except tf.errors.OutOfRangeError:
                    print("End of epoch %d" % i)
                    break


if __name__ == '__main__':
    main()
