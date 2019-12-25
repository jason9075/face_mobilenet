import argparse
import glob
import logging
import logging.handlers as handlers
import os
import random
import timeit
from datetime import datetime

import numpy as np
import tensorflow as tf

import utils
from backend.loss_function import triplet_loss
from backend.net_builder import NetBuilder, Arch, FinalLayer

MODEL_OUT_PATH = os.path.join('model_out')
INPUT_SIZE = (112, 112)
LR_STEPS = [60000, 120000, 160000]
LR_VAL = [0.01, 0.005, 0.001, 0.0005]
ACC_LOW_BOUND = 0.85
EPOCH = 10000
SAVER_MAX_KEEP = 5
MOMENTUM = 0.9
MODEL = Arch.RES_NET50

MIN_IMAGES_PER_PERSON = 4
PAIR_PER_PERSON = MIN_IMAGES_PER_PERSON - 1
BATCH_SIZE = 40  # is must be even number
STUDY_SIZE = BATCH_SIZE * 6
STRATEGY = 'hard'
EMBEDDING_SIZE = 128
ALPHA = 0.5  # Positive to negative triplet distance margin.

SHOW_INFO_INTERVAL = 100
SUMMARY_INTERVAL = 2000
CKPT_INTERVAL = 1000
VALIDATE_INTERVAL = 2000


class ImageClass:
    def __init__(self, name, image_paths):
        self.name = name
        self.image_paths = image_paths

    def __str__(self):
        return self.name + ', ' + str(len(self.image_paths)) + ' images'

    def __len__(self):
        return len(self.image_paths)


class TripletPair:
    def __init__(self, a, p, n):
        self.a = a
        self.p = p
        self.n = n

    def __lt__(self, o):
        return True

    def __gt__(self, o):
        return True

    def to_path(self):
        return [self.a, self.p, self.n]


def purge():
    for f in glob.glob(os.path.join('events/events*')):
        os.remove(f)


def get_parser():
    parser = argparse.ArgumentParser(description='parameters to train net')
    parser.add_argument('--data-dir', default='./images/star224/', help='training data path')
    parser.add_argument('--pretrain', default='', help='pretrain model ckpt, ex: MDL_iter_1110000.ckpt')
    args = parser.parse_args()
    return args


def init_log():
    global logger
    filename = os.path.join('log',
                            datetime.now().strftime("%Y-%m-%d_%H-%M-%S.log"))

    logger = logging.getLogger('MAIN')
    logger.propagate = False
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


def study_buffer(buffer_list, size):
    tp_list = []
    for _ in range(size):
        tp_list.append(buffer_list.pop())

    return tp_list


def random_batch(buffer_list, size):
    study_sample = study_buffer(buffer_list, size)
    study_sample = [sample.to_path() for sample in study_sample]
    study_sample = list(map(list, zip(*study_sample)))  # transpose
    return [y for x in study_sample for y in x]  # flatten


def hard_batch(sess, l2_embedding, path_node, is_training_node, buffer_list, study_size, batch_size):
    study_sample = study_buffer(buffer_list, study_size)
    sample_loss = np.zeros(study_size)

    for idx in range(0, study_size, batch_size):
        sample = study_sample[idx:idx + batch_size]
        sample = [tp.to_path() for tp in sample]
        sample = list(map(list, zip(*sample)))  # transpose
        sample = [y for x in sample for y in x]

        feed_dict = {
            path_node: sample,
            is_training_node: False
        }
        emb = sess.run(l2_embedding, feed_dict=feed_dict)
        a_emb, p_emb, n_emb = np.split(emb, 3)

        p_dist = np.sum(np.square(a_emb - p_emb), axis=1)
        n_dist = np.sum(np.square(a_emb - n_emb), axis=1)
        sample_loss[idx:idx + batch_size] = p_dist - n_dist

    _, study_sample = zip(*sorted(zip(sample_loss, study_sample), reverse=True))

    half_size = int(batch_size / 2)

    hard_sample = study_sample[:half_size]
    random_sample = np.random.choice(study_sample[half_size:], size=half_size, replace=False)
    mix_sample = [item for sublist in zip(hard_sample, random_sample) for item in sublist]

    mix_sample = [sample.to_path() for sample in mix_sample]
    mix_sample = list(map(list, zip(*mix_sample)))  # transpose
    return [y for x in mix_sample for y in x]  # flatten


def main():
    args = get_parser()

    purge()
    init_log()

    builder = NetBuilder()

    dataset = get_dataset(args.data_dir)
    total_images_cnt = int(np.sum([len(item) for item in dataset]))
    log('Total %d \'s person with %d images.' % (len(dataset), total_images_cnt))
    log('lr values:{}'.format(LR_VAL))
    log('lr steps:{}'.format(LR_STEPS))

    verification_path = os.path.join('tfrecord', 'verification.tfrecord')
    ver_dataset = utils.get_ver_data(verification_path, INPUT_SIZE)

    input_layer = tf.placeholder(
        name='input_images',
        shape=[None, INPUT_SIZE[0], INPUT_SIZE[1], 3],
        dtype=tf.float32)
    is_training = tf.placeholder_with_default(False, (), name='is_training')

    image_paths_placeholder = tf.placeholder(tf.string, shape=(None,), name='image_paths')

    triplet_input = triplet_image_process(image_paths_placeholder)

    with tf.name_scope('train'):
        train_net = builder.input_and_train_node(triplet_input, is_training) \
            .arch_type(MODEL) \
            .final_layer_type(FinalLayer.G) \
            .build()

    with tf.name_scope('valid'):
        val_net = builder.input_and_train_node(input_layer, tf.constant(False)) \
            .arch_type(MODEL) \
            .final_layer_type(FinalLayer.G) \
            .build(reuse=True)

    with tf.variable_scope('loss'):
        anchor, positive, negative = tf.split(train_net, 3)

        inference_loss, fail_count = triplet_loss(anchor, positive, negative, ALPHA)
        wd_loss = tf.reduce_sum(
            tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES), name='wd_loss')
        total_loss = tf.add(inference_loss, wd_loss, name='total_loss')

    with tf.variable_scope('etc'):
        global_step = tf.train.get_or_create_global_step()
        lr = tf.train.piecewise_constant(
            global_step,
            boundaries=LR_STEPS,
            values=LR_VAL,
            name='lr_schedule')

        opt = tf.train.MomentumOptimizer(learning_rate=lr, momentum=MOMENTUM)
        grads = opt.compute_gradients(total_loss)

    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(update_ops):
        train_op = opt.apply_gradients(grads, global_step=global_step)

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True

    total_parameters = 0
    for variable in tf.trainable_variables():
        shape = variable.get_shape()
        variable_parameters = 1
        for dim in shape:
            variable_parameters *= dim.value
        total_parameters += variable_parameters
    log('Total parameters count: %d' % total_parameters)

    with tf.Session(config=config) as sess:

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
        summaries.append(tf.summary.scalar('fail_count', fail_count))
        summary_op = tf.summary.merge(summaries)
        saver = tf.train.Saver(max_to_keep=SAVER_MAX_KEEP)

        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())

        if args.pretrain != '':
            restore_saver = tf.train.Saver()
            restore_saver.restore(sess,
                                  os.path.join(MODEL_OUT_PATH, args.pretrain))

        have_best = False
        best_accuracy = 0
        step = 1
        try:
            for epoch_idx in range(EPOCH):
                batch_idx = 1
                buffer_list = sample_buffer(dataset, PAIR_PER_PERSON)
                if STRATEGY == 'hard':
                    epoch_size = int(len(buffer_list) / STUDY_SIZE)
                else:
                    epoch_size = int(len(buffer_list) / BATCH_SIZE)
                while batch_idx <= epoch_size:
                    # Select

                    if STRATEGY == 'hard':
                        sample = hard_batch(sess, train_net, image_paths_placeholder, is_training, buffer_list,
                                            STUDY_SIZE, BATCH_SIZE)
                    else:
                        sample = random_batch(buffer_list, BATCH_SIZE)

                    # Training
                    run_dict = {
                        'train_op': train_op,
                        'global_step': global_step
                    }
                    feed_dict = {
                        image_paths_placeholder: sample,
                        is_training: True
                    }

                    if step % SHOW_INFO_INTERVAL == 0:
                        run_dict['total_loss'] = total_loss
                        run_dict['inference_loss'] = inference_loss
                        run_dict['wd_loss'] = wd_loss
                        run_dict['fail_count'] = fail_count

                    if step % SUMMARY_INTERVAL == 0:
                        run_dict['summary'] = summary_op

                    start_time = timeit.default_timer()
                    results = sess.run(run_dict, feed_dict=feed_dict)
                    duration = timeit.default_timer() - start_time

                    # print training information
                    if step % SHOW_INFO_INTERVAL == 0:
                        show_info(epoch_idx, batch_idx, epoch_size, step, duration, results)

                    # save summary
                    if step % SUMMARY_INTERVAL == 0:
                        save_summary(summary, results)

                    # save ckpt files
                    if step % CKPT_INTERVAL == 0 and not have_best:
                        save_ckpt(step, saver, sess)

                    # validate
                    if step % VALIDATE_INTERVAL == 0:
                        best_accuracy = validate(best_accuracy, step, summary,
                                                 input_layer, val_net, saver, sess, ver_dataset)
                    batch_idx += 1
                    step += 1
        except Exception as err:
            log('Exception, saving interrupt ckpt. err: {}'.format(err))
            filename = '{:s}_err_iter_{:d}.ckpt'.format(MODEL.name, step)
            filename = os.path.join(MODEL_OUT_PATH, filename)
            saver.save(sess, filename)
            raise err


def triplet_image_process(image_paths_placeholder):
    def _parse(image_path):
        file_contents = tf.read_file(image_path)
        image = tf.image.decode_image(file_contents, channels=3)

        image = tf.image.random_brightness(image, 0.2)
        image = tf.image.random_saturation(image, 0.6, 1.6)
        image = tf.image.random_contrast(image, 0.6, 1.4)
        image = tf.image.random_flip_left_right(image)

        # pylint: disable=no-member
        image.set_shape((INPUT_SIZE[0], INPUT_SIZE[1], 3))
        image = utils.tf_pre_process_image(image, INPUT_SIZE)

        return image

    with tf.variable_scope('triplet_image_process'):
        return tf.map_fn(_parse, image_paths_placeholder, dtype=tf.float32)


def validate(best_accuracy, step, summary_writer, input_layer, net, saver, sess,
             ver_dataset):
    val_acc, val_thr = utils.ver_test(
        data_set=ver_dataset,
        sess=sess,
        l2_embedding_tensor=net,
        input_placeholder=input_layer)
    log('Test accuracy is: {}, thr: {}'.format(val_acc, val_thr))
    summary = tf.Summary()
    summary.value.add(tag='test/accuracy', simple_value=np.mean(val_acc))
    summary.value.add(tag='test/thr', simple_value=val_thr)
    summary_writer.add_summary(summary, step)
    if ACC_LOW_BOUND < val_acc and best_accuracy < val_acc:
        log('Best accuracy is %.5f' % val_acc)
        filename = '{:s}_best_{:.5f}_iter_{:d}.ckpt'.format(MODEL.name, val_acc, step)
        filename = os.path.join(MODEL_OUT_PATH, filename)
        saver.save(sess, filename)
        return val_acc
    return best_accuracy


def save_ckpt(step, saver, sess):
    log('Step: %d, saving ckpt.' % step)
    filename = '{:s}_iter_{:d}.ckpt'.format(MODEL.name, step)
    filename = os.path.join(MODEL_OUT_PATH, filename)
    saver.save(sess, filename)


def save_summary(summary, results):
    summary.add_summary(results['summary'], results['global_step'])


def show_info(epoch_idx, batch_idx, epoch_size, step, duration, results):
    log('Epoch: [%d][%d/%d], step %d, total_loss: %.2f, inf_loss: %.2f, weight_loss: '
        '%.2f, time %.3f sec, fail rate: %.3f' %
        (epoch_idx, batch_idx, epoch_size, step, results['total_loss'], results['inference_loss'],
         results['wd_loss'], duration, results['fail_count'] / BATCH_SIZE))


def sample_buffer(dataset, pair_per_person):
    buffer_sample = []
    np.random.shuffle(dataset)

    for idx, ic in enumerate(dataset):
        selection = ic.image_paths.copy()
        for _ in range(pair_per_person):
            a_path = random.choice(selection)
            selection.remove(a_path)
            p_path = random.choice(selection)
            n_path = random.choice(dataset[idx - 1].image_paths)
            buffer_sample.append(TripletPair(a_path, p_path, n_path))

    np.random.shuffle(buffer_sample)
    return buffer_sample


def get_dataset(path):
    dataset = []
    path_exp = os.path.expanduser(path)
    classes = [path for path in os.listdir(path_exp) \
               if os.path.isdir(os.path.join(path_exp, path))]
    classes.sort()
    nrof_classes = len(classes)
    for i in range(nrof_classes):
        class_name = classes[i]
        facedir = os.path.join(path_exp, class_name)
        image_paths = get_image_paths(facedir)
        dataset.append(ImageClass(class_name, image_paths))

    dataset = [data for data in dataset if MIN_IMAGES_PER_PERSON <= len(data)]
    return dataset


def get_image_paths(facedir):
    image_paths = []
    if os.path.isdir(facedir):
        images = os.listdir(facedir)
        image_paths = [os.path.join(facedir, img) for img in images]
    return image_paths


if __name__ == '__main__':
    main()
