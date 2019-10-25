import argparse
import glob
import logging
import logging.handlers as handlers
import os
import pickle
import time
from datetime import datetime

import cv2
# import mxnet as mx
import numpy as np
import tensorflow as tf
from sklearn import preprocessing
from tensorflow.core.protobuf import config_pb2

import utils
from backend.loss_function import combine_loss_val
from backend.net_builder import NetBuilder, Arch, FinalLayer

MODEL_OUT_PATH = os.path.join('model_out')
INPUT_SIZE = (224, 224)
LR_STEPS = [80000, 120000, 160000]
ACC_LOW_BOUND = 0.85
NUM_CLASSES = 1037
BATCH_SIZE = 32
BUFFER_SIZE = 500
EPOCH = 10000
SAVER_MAX_KEEP = 5
MOMENTUM = 0.9
M1 = 1.0
M2 = 0.1
M3 = 0.1
SCALE = 64

SHOW_INFO_INTERVAL = 100
SUMMARY_INTERVAL = 300
CKPT_INTERVAL = 1000
VALIDATE_INTERVAL = 2000
MONITOR_NODE = ''


def purge():
    for f in glob.glob(os.path.join('events/events*')):
        os.remove(f)


def get_parser():
    parser = argparse.ArgumentParser(description='parameters to train net')
    parser.add_argument('--pretrain', default='', help='pretrain model ckpt, ex: InsightFace_iter_1110000.ckpt')
    args = parser.parse_args()
    return args


def init_log():
    global logger
    filename = os.path.join('log',
                            datetime.now().strftime("%Y-%m-%d_%H-%M-%S.log"))

    logger = logging.getLogger('MAIN')
    logger.setLevel(logging.DEBUG)
    log_handler = handlers.TimedRotatingFileHandler(
        filename, when='D', interval=1, backupCount=30)
    log_handler.setLevel(logging.DEBUG)
    formatter = logging.Formatter(
        '%(asctime)s %(name)-12s %(levelname)-8s %(message)s')
    log_handler.setFormatter(formatter)
    logger.addHandler(log_handler)


def log(msg, verbose=True):
    if verbose:
        print(msg)
    logger.info(msg)


def main():
    args = get_parser()

    purge()
    init_log()

    builder = NetBuilder()

    record_path = os.path.join('tfrecord', 'train.tfrecord')
    data_set = tf.data.TFRecordDataset(record_path)
    data_set = data_set.map(utils.parse_function)
    data_set = data_set.shuffle(buffer_size=BUFFER_SIZE)
    data_set = data_set.batch(BATCH_SIZE)
    iterator = data_set.make_initializable_iterator()
    next_element = iterator.get_next()

    verification_path = os.path.join('tfrecord', 'verification.tfrecord')
    ver_dataset = utils.get_ver_data(verification_path)

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True

    with tf.Session(config=config) as sess:

        global_step = tf.Variable(
            name='global_step', initial_value=0, trainable=False)
        input_layer = tf.placeholder(
            name='input_images',
            shape=[None, INPUT_SIZE[0], INPUT_SIZE[1], 3],
            dtype=tf.float32)
        labels = tf.placeholder(
            name='img_labels', shape=[
                None,
            ], dtype=tf.int64)
        is_training = tf.placeholder_with_default(False, (), name='is_training')

        net = builder.input_and_train_node(input_layer, is_training) \
            .arch_type(Arch.RES_NET50) \
            .final_layer_type(FinalLayer.G) \
            .build()

        logit = combine_loss_val(
            embedding=net.embedding,
            gt_labels=labels,
            num_labels=NUM_CLASSES,
            batch_size=BATCH_SIZE,
            m1=M1,
            m2=M2,
            m3=M3,
            s=SCALE)

        inference_loss = tf.reduce_mean(
            tf.nn.sparse_softmax_cross_entropy_with_logits(
                logits=logit, labels=labels), name='inference_loss')
        wd_loss = tf.reduce_sum(
            tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES), name='wd_loss')
        total_loss = tf.add(inference_loss, wd_loss, name='total_loss')

        log('lr_steps:{}'.format(LR_STEPS))
        lr = tf.train.piecewise_constant(
            global_step,
            boundaries=LR_STEPS,
            values=[0.001, 0.0005, 0.0003, 0.0001],
            name='lr_schedule')

        opt = tf.train.AdamOptimizer(learning_rate=lr, beta1=0.9, beta2=0.995)
        grads = opt.compute_gradients(total_loss)

        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            train_op = opt.apply_gradients(grads, global_step=global_step)

        pred = tf.nn.softmax(logit)
        acc = tf.reduce_mean(
            tf.cast(
                tf.equal(tf.argmax(pred, axis=1), labels), dtype=tf.float32))

        summary = tf.summary.FileWriter('events/', sess.graph)
        summaries = []

        for grad, var in grads:
            if grad is not None:
                summaries.append(
                    tf.summary.histogram(var.op.name + '/gradients', grad))

        for var in tf.trainable_variables():
            summaries.append(tf.summary.histogram(var.op.name, var))

        summaries.append(tf.summary.scalar('loss/inference', inference_loss))
        summaries.append(tf.summary.scalar('loss/weight_decay', wd_loss))
        summaries.append(tf.summary.scalar('loss/total', total_loss))
        summaries.append(tf.summary.scalar('learning_rate', lr))
        summaries.append(tf.summary.scalar('accuracy', acc))
        summary_op = tf.summary.merge(summaries)
        saver = tf.train.Saver(max_to_keep=SAVER_MAX_KEEP)

        total_parameters = 0
        for variable in tf.trainable_variables():
            shape = variable.get_shape()
            variable_parameters = 1
            for dim in shape:
                variable_parameters *= dim.value
            total_parameters += variable_parameters
        print('total parameters count: %d' % total_parameters)

        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())

        if args.pretrain != '':
            restore_saver = tf.train.Saver()
            restore_saver.restore(sess,
                                  os.path.join(MODEL_OUT_PATH, args.pretrain))

        count = 0
        have_best = False
        best_accuracy = 0
        for i in range(EPOCH):
            sess.run(iterator.initializer)
            while True:
                try:
                    images_train, labels_train = sess.run(next_element)
                    if images_train.shape[0] != BATCH_SIZE:
                        break
                    feed_dict = {
                        input_layer: images_train,
                        labels: labels_train,
                        is_training: True
                    }
                    start = time.time()
                    _, total_loss_val, inference_loss_val, wd_loss_val, acc_val = \
                        sess.run([train_op, total_loss, inference_loss, wd_loss, acc],
                                 feed_dict=feed_dict,
                                 options=config_pb2.RunOptions(report_tensor_allocations_upon_oom=True))
                    if MONITOR_NODE != '':
                        mon_dict = {
                            input_layer: images_train,
                            labels: labels_train,
                            is_training: False
                        }
                        mon_node = tf.get_default_graph().get_tensor_by_name(MONITOR_NODE)
                        node_v = sess.run(mon_node, feed_dict=mon_dict)
                        log('{} max value: {}'.format(MONITOR_NODE, np.max(node_v)), verbose=False)
                    end = time.time()
                    pre_sec = BATCH_SIZE / (end - start)
                    if count == 0:
                        count += 1
                        continue
                    # print training information
                    if count % SHOW_INFO_INTERVAL == 0:
                        show_info(acc_val, count, i, images_train,
                                  inference_loss_val, input_layer,
                                  labels_train, net, pre_sec, sess,
                                  total_loss_val, is_training, wd_loss_val)
                    # save summary
                    if count % SUMMARY_INTERVAL == 0:
                        save_summary(count, images_train, input_layer, labels,
                                     labels_train, sess, summary, summary_op,
                                     is_training)

                    # save ckpt files
                    if count % CKPT_INTERVAL == 0 and not have_best:
                        save_ckpt(count, i, saver, sess)

                    # validate
                    if count % VALIDATE_INTERVAL == 0:
                        best_accuracy = validate(best_accuracy, count,
                                                 input_layer, net, saver, sess,
                                                 is_training, ver_dataset)

                    count += 1
                except tf.errors.OutOfRangeError:
                    log("End of epoch %d" % i)
                    break
                except Exception as err:
                    log('Exception, saving ckpt. err: {}'.format(err))
                    filename = 'InsightFace_iter_err_{:d}'.format(
                        count) + '.ckpt'
                    filename = os.path.join(MODEL_OUT_PATH, filename)
                    saver.save(sess, filename)
                    raise err


def validate(best_accuracy, count, input_layer, net, saver, sess, is_training,
             ver_dataset):
    feed_dict_test = {is_training: False}
    val_acc, val_thr = utils.ver_test(
        data_set=ver_dataset,
        sess=sess,
        embedding_tensor=net.embedding,
        feed_dict=feed_dict_test,
        input_placeholder=input_layer)
    log('test accuracy is: {}, thr: {}'.format(val_acc, val_thr))
    if ACC_LOW_BOUND < val_acc and best_accuracy < val_acc:
        log('best accuracy is %.5f' % val_acc)
        filename = 'InsightFace_iter_best_{:.2f}_{:d}'.format(val_acc, count) + '.ckpt'
        filename = os.path.join(MODEL_OUT_PATH, filename)
        saver.save(sess, filename)
        return val_acc
    return best_accuracy


def save_ckpt(count, i, saver, sess):
    log('epoch: %d,count: %d, saving ckpt.' % (i, count))
    filename = 'InsightFace_iter_{:d}'.format(count) + '.ckpt'
    filename = os.path.join(MODEL_OUT_PATH, filename)
    saver.save(sess, filename)


def save_summary(count, images_train, input_layer, labels, labels_train, sess,
                 summary, summary_op, is_training):
    feed_summary_dict = {
        input_layer: images_train,
        labels: labels_train,
        is_training: False
    }
    summary_op_val = sess.run(summary_op, feed_dict=feed_summary_dict)
    summary.add_summary(summary_op_val, count)


def show_info(acc_val, count, i, images_train, inference_loss_val, input_layer,
              labels_train, net, pre_sec, sess, total_loss_val, is_training,
              wd_loss_val):
    log('epoch %d, total_step %d, total loss is %.2f , inference loss is %.2f, weight deacy '
        'loss is %.2f, training accuracy is %.6f, time %.3f samples/sec' %
        (i, count, total_loss_val, inference_loss_val, wd_loss_val, acc_val,
         pre_sec))
    feed_dict = {input_layer: images_train[:2], is_training: False}
    embedding_pair = sess.run(net.embedding, feed_dict=feed_dict)
    vector_pair = preprocessing.normalize(
        [embedding_pair[0], embedding_pair[1]])
    dist = np.linalg.norm(vector_pair[0] - vector_pair[1])
    log('(%d vs %d)distance: %.2f' % (labels_train[0], labels_train[1], dist))


def load_bin(bin_path):
    bins, issame_list = pickle.load(open(bin_path, 'rb'), encoding='bytes')
    data_list = []
    for _ in [0, 1]:
        data = np.empty((len(issame_list) * 2, INPUT_SIZE[0], INPUT_SIZE[1], 3))
        data_list.append(data)
    for i in range(len(issame_list) * 2):
        _bin = bins[i]
        img = mx.image.imdecode(_bin).asnumpy()
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        for flip in [0, 1]:
            if flip == 1:
                img = np.fliplr(img)
            data_list[flip][i, ...] = img
        i += 1
        if i % 1000 == 0:
            print('loading bin', i)
    print(data_list[0].shape)
    return data_list, issame_list


def test():
    verification_path = os.path.join('tfrecord', 'verification.tfrecord')
    ver_dataset = utils.get_ver_data(verification_path)

    # lfw_set = load_bin('tfrecord/lfw.bin')

    with tf.Session() as sess:
        saver = tf.train.import_meta_graph(
            'model_out/InsightFace_iter_198000.ckpt.meta', clear_devices=True)
        saver.restore(sess, "model_out/InsightFace_iter_198000.ckpt")

        # image1 = cv2.imread('images/image_db/andy/gen_3791a1_21.jpg')
        # image2 = cv2.imread('images/image_db/andy/gen_3791a1_13.jpg')
        #
        # image1 = processing(image1)
        # image2 = processing(image2)

        input_tensor = tf.get_default_graph().get_tensor_by_name(
            "input_images:0")
        trainable = tf.get_default_graph().get_tensor_by_name("trainable_bn:0")
        embedding_tensor = tf.get_default_graph().get_tensor_by_name(
            "gdc/embedding/Identity:0")

        feed_dict_test = {trainable: False}
        val_acc, val_thr = utils.ver_test(
            data_set=ver_dataset,
            sess=sess,
            embedding_tensor=embedding_tensor,
            feed_dict=feed_dict_test,
            input_placeholder=input_tensor)

        print('astra acc: %.2f, thr: %.2f' % (val_acc, val_thr))

        # val_acc, val_thr = utils.lfw_test(
        #     data_set=lfw_set,
        #     sess=sess,
        #     embedding_tensor=embedding_tensor,
        #     feed_dict=feed_dict_test,
        #     input_placeholder=input_tensor)
        #
        # print('lfw acc: %.2f, thr: %.2f' % (val_acc, val_thr))

        # feed_dict = {input_tensor: np.expand_dims(image1, 0), trainable: False}
        # vector1 = sess.run(embedding_tensor, feed_dict=feed_dict)
        #
        # feed_dict = {input_tensor: np.expand_dims(image2, 0), trainable: False}
        # vector2 = sess.run(embedding_tensor, feed_dict=feed_dict)
        #
        # vector1 = preprocessing.normalize(vector1)
        # vector2 = preprocessing.normalize(vector2)
        #
        # print(vector1)
        # print(vector2)
        #
        # print('dist: ',np.linalg.norm(vector1 - vector2))


def processing(img):
    img = cv2.resize(img, (224, 224))
    # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = np.array(img) - 127.5
    img *= 0.0078125
    return img


if __name__ == '__main__':
    main()
    # test()
