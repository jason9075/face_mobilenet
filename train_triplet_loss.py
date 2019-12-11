import argparse
import glob
import logging
import logging.handlers as handlers
import os
import random
import time
from datetime import datetime

import numpy as np
import tensorflow as tf

import utils
from backend.loss_function import triplet_loss
from backend.net_builder import NetBuilder, Arch, FinalLayer

MODEL_OUT_PATH = os.path.join('model_out')
INPUT_SIZE = (112, 112)
LR_STEPS = [80000, 120000, 160000]
LR_VAL = [0.01, 0.005, 0.001, 0.0005]
ACC_LOW_BOUND = 0.85
EPOCH = 10000
SAVER_MAX_KEEP = 5
MOMENTUM = 0.9
MODEL = Arch.RES_NET50

MIN_IMAGES_PER_PERSON = 4
PAIR_PER_PERSON = MIN_IMAGES_PER_PERSON - 1
BATCH_SIZE = 32
STUDY_SIZE = BATCH_SIZE * 1
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
        self.loss = 0

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
    study_sample = study_buffer(buffer_list, STUDY_SIZE)[:size]
    study_sample = [sample.to_path() for sample in study_sample]
    study_sample = list(map(list, zip(*study_sample)))  # transpose
    return [y for x in study_sample for y in x]  # flatten


def hard_batch(buffer_list, size):
    study_sample = study_buffer(buffer_list, STUDY_SIZE)


def main():
    args = get_parser()

    purge()
    init_log()

    builder = NetBuilder()

    dataset = get_dataset(args.data_dir)
    total_images_cnt = np.sum([len(item) for item in dataset])
    print('total image count: %d' % total_images_cnt)

    verification_path = os.path.join('tfrecord', 'verification.tfrecord')
    ver_dataset = utils.get_ver_data(verification_path, INPUT_SIZE)

    with tf.Graph().as_default():

        global_step = tf.Variable(
            name='global_step', initial_value=0, trainable=False)
        input_layer = tf.placeholder(
            name='input_images',
            shape=[None, INPUT_SIZE[0], INPUT_SIZE[1], 3],
            dtype=tf.float32)
        is_training = tf.placeholder_with_default(False, (), name='is_training')

        image_paths_placeholder = tf.placeholder(tf.string, shape=(None,), name='image_paths')

        triplet_input = triplet_image_process(image_paths_placeholder)

        with tf.name_scope('train_net'):
            train_net = builder.input_and_train_node(triplet_input, is_training) \
                .arch_type(MODEL) \
                .final_layer_type(FinalLayer.G) \
                .build()

        # with tf.name_scope('valid_net'):
        #     val_net = builder.input_and_train_node(input_layer, is_training) \
        #         .arch_type(MODEL) \
        #         .final_layer_type(FinalLayer.G) \
        #         .build(reuse=True)

        l2_embeddings = tf.nn.l2_normalize(train_net.embedding, axis=1, epsilon=1e-10, name='l2_embeddings')
        anchor, positive, negative = tf.split(l2_embeddings, 3)

        inference_loss, fail_count = triplet_loss(anchor, positive, negative, ALPHA)
        wd_loss = tf.reduce_sum(
            tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES), name='wd_loss')
        total_loss = tf.add(inference_loss, wd_loss, name='total_loss')

        log('lr_steps:{}'.format(LR_STEPS))
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

        with tf.Session(config=config) as sess:

            summary = tf.summary.FileWriter('events/', sess.graph)
            coord = tf.train.Coordinator()
            tf.train.start_queue_runners(coord=coord, sess=sess)
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
            summaries.append(tf.summary.scalar('leraning_rate', lr))
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

            have_best = False
            best_accuracy = 0
            step = 1
            try:
                for epoch_idx in range(EPOCH):
                    batch_idx = 1
                    buffer_list = sample_buffer(dataset, PAIR_PER_PERSON)
                    epoch_size = len(buffer_list) / BATCH_SIZE
                    while batch_idx < epoch_size:
                        # Select
                        # sample = hard_batch(buffer_list, BATCH_SIZE)
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

                        start_time = time.time()
                        results = sess.run(run_dict, feed_dict=feed_dict)
                        duration = time.time() - start_time

                        # print training information
                        if step % SHOW_INFO_INTERVAL == 0:
                            show_info(epoch_idx, batch_idx, epoch_size, step, duration, results)

                        # save summary
                        if step % SUMMARY_INTERVAL == 0:
                            save_summary(summary, results)

                        # save ckpt files
                        if step % CKPT_INTERVAL == 0 and not have_best:
                            save_ckpt(step, epoch_idx, saver, sess)

                        # # validate
                        # if step % VALIDATE_INTERVAL == 0:
                        #     best_accuracy = validate(best_accuracy, step,
                        #                              input_layer, net, saver, sess,
                        #                              is_training, ver_dataset)
                        batch_idx += 1
                        step +=1
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


def validate(best_accuracy, step, input_layer, net, saver, sess, is_training,
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
        filename = '{:s}_best_{:.5f}_iter_{:d}.ckpt'.format(MODEL.name, val_acc, step)
        filename = os.path.join(MODEL_OUT_PATH, filename)
        saver.save(sess, filename)
        return val_acc
    return best_accuracy


def save_ckpt(step, i, saver, sess):
    log('epoch: %d, step: %d, saving ckpt.' % (i, step))
    filename = '{:s}_iter_{:d}.ckpt'.format(MODEL.name, step)
    filename = os.path.join(MODEL_OUT_PATH, filename)
    saver.save(sess, filename)


def save_summary(summary, results):
    summary.add_summary(results['summary'], results['global_step'])


def show_info(epoch_idx, batch_idx, epoch_size, step, duration, results):
    log('Epoch: [%d][%d/%d], step %d, total_loss: %.2f, inf_loss: %.2f, weight_loss: '
        '%.2f, Time %.3f, Fail Rate: %.3f' %
        (epoch_idx, batch_idx, epoch_size, step, results['total_loss'], results['inference_loss'],
         results['wd_loss'], duration, results['fail_count']/BATCH_SIZE))


def sample_buffer(dataset, pair_per_person):
    buffer_sample = []
    np.random.shuffle(dataset)

    for idx, ic in enumerate(dataset):
        selection = ic.image_paths
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


def select_triplets(embeddings, nrof_images_per_class, image_paths, people_per_batch, alpha):
    trip_idx = 0
    emb_start_idx = 0
    num_trips = 0
    triplets = []

    for i in range(people_per_batch):
        nrof_images = int(nrof_images_per_class[i])
        for j in range(1, nrof_images):
            a_idx = emb_start_idx + j - 1
            neg_dists_sqr = np.sum(np.square(embeddings[a_idx] - embeddings), 1)
            for pair in range(j, nrof_images):  # For every possible positive pair.
                p_idx = emb_start_idx + pair
                pos_dist_sqr = np.sum(np.square(embeddings[a_idx] - embeddings[p_idx]))
                neg_dists_sqr[emb_start_idx:emb_start_idx + nrof_images] = np.NaN
                all_neg = np.where(np.logical_and(neg_dists_sqr - pos_dist_sqr < alpha, pos_dist_sqr < neg_dists_sqr))[
                    0]  # FaceNet selection
                # all_neg = np.where(neg_dists_sqr - pos_dist_sqr < alpha)[0]  # VGG Face selection
                nrof_random_negs = all_neg.shape[0]
                if 0 < nrof_random_negs:
                    rnd_idx = np.random.randint(nrof_random_negs)
                    n_idx = all_neg[rnd_idx]
                    triplets.append((image_paths[a_idx], image_paths[p_idx], image_paths[n_idx]))
                    trip_idx += 1

                num_trips += 1

        emb_start_idx += nrof_images

    np.random.shuffle(triplets)
    return triplets, num_trips, len(triplets)


if __name__ == '__main__':
    main()
