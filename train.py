import os
import time

import tensorflow as tf

import utils
from backend.loss_function import combine_loss_val
from backend.mobilenet_v2 import MobileNetV2


def main():
    num_classes = 10
    labels = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    batch_size = 2
    buffer_size = 10
    labels = labels[:batch_size]
    epoch = 100
    lr = 0.001
    saver_maxkeep = 5
    momentum = 0.99
    show_info_interval = 20
    summary_interval = 300

    record_path = os.path.join('tfrecord', 'train.tfrecord')
    data_set = tf.data.TFRecordDataset(record_path)
    data_set = data_set.map(utils.parse_function)
    data_set = data_set.shuffle(buffer_size=buffer_size)
    data_set = data_set.batch(batch_size)
    iterator = data_set.make_initializable_iterator()
    next_element = iterator.get_next()

    with tf.Session() as sess:
        summary = tf.summary.FileWriter('events/', sess.graph)
        summaries = []

        global_step = tf.Variable(name='global_step', initial_value=0, trainable=False)

        net = MobileNetV2(input_size=(224, 224))

        logit = combine_loss_val(embedding=net.embedding, gt_labels=labels, num_labels=num_classes,
                                 batch_size=batch_size, m1=1, m2=0, m3=0, s=64)
        inference_loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logit, labels=labels))

        wd_loss = 0
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
        saver = tf.train.Saver(max_to_keep=saver_maxkeep)
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
                    feed_dict = {images: images_train, labels: labels_train, dropout_rate: 0.4}
                    feed_dict.update(net.all_drop)
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
                        feed_dict = {images: images_train, labels: labels_train}
                        feed_dict.update(net.all_drop)
                        summary_op_val = sess.run(summary_op, feed_dict=feed_dict)
                        summary.add_summary(summary_op_val, count)

                    # save ckpt files
                    if count > 0 and count % args.ckpt_interval == 0:
                        filename = 'InsightFace_iter_{:d}'.format(count) + '.ckpt'
                        filename = os.path.join(args.ckpt_path, filename)
                        saver.save(sess, filename)

                    # validate
                    if count > 0 and count % args.validate_interval == 0:
                        feed_dict_test = {dropout_rate: 1.0}
                        feed_dict_test.update(tl.utils.dict_to_one(net.all_drop))
                        results = ver_test(ver_list=ver_list, ver_name_list=ver_name_list, nbatch=count, sess=sess,
                                           embedding_tensor=embedding_tensor, batch_size=batch_size,
                                           feed_dict=feed_dict_test,
                                           input_placeholder=images)
                        print('test accuracy is: ', str(results[0]))
                        total_accuracy[str(count)] = results[0]
                        if max(results) > 0.996:
                            print('best accuracy is %.5f' % max(results))
                            filename = 'InsightFace_iter_best_{:d}'.format(count) + '.ckpt'
                            filename = os.path.join(args.ckpt_path, filename)
                            saver.save(sess, filename)
                except tf.errors.OutOfRangeError:
                    print("End of epoch %d" % i)
                    break







if __name__ == '__main__':
    main()
