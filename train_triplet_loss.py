import argparse
import glob
import itertools
import logging
import logging.handlers as handlers
import os
import time
from datetime import datetime

import numpy as np
import tensorflow as tf
from tensorflow.python.ops import data_flow_ops

import utils
from backend.loss_function import triplet_loss
from backend.net_builder import NetBuilder, Arch, FinalLayer

MODEL_OUT_PATH = os.path.join('model_out')
INPUT_SIZE = (224, 224)
LR_STEPS = [80000, 120000, 160000]
LR_VAL = [0.01, 0.005, 0.001, 0.0005]
ACC_LOW_BOUND = 0.85
EPOCH = 10000
SAVER_MAX_KEEP = 5
MOMENTUM = 0.9
MODEL = Arch.RES_NET50

PEOPLE_PER_BATCH = 45  # must be divisible by 3
IMAGES_PER_PERSON = 10
BATCH_SIZE = 45  # must be divisible by 3
EMBEDDING_SIZE = 128
NUM_PREPROCESS_THREADS = 4
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


def main():
    args = get_parser()

    purge()
    init_log()

    builder = NetBuilder()

    dataset = get_dataset(args.data_dir)
    total_images_cnt = np.sum([len(item) for item in dataset])
    epoch_size = total_images_cnt // BATCH_SIZE

    verification_path = os.path.join('tfrecord', 'verification.tfrecord')
    ver_dataset = utils.get_ver_data(verification_path)

    with tf.Graph().as_default():

        global_step = tf.Variable(
            name='global_step', initial_value=0, trainable=False)
        input_layer = tf.placeholder(
            name='input_images',
            shape=[None, INPUT_SIZE[0], INPUT_SIZE[1], 3],
            dtype=tf.float32)
        is_training = tf.placeholder_with_default(False, (), name='is_training')

        batch_size_placeholder = tf.placeholder(tf.int32, name='batch_size')
        image_paths_placeholder = tf.placeholder(tf.string, shape=(None, 3), name='image_paths')
        labels_placeholder = tf.placeholder(tf.int64, shape=(None, 3), name='labels')

        input_queue = data_flow_ops.FIFOQueue(capacity=100000,
                                              dtypes=[tf.string, tf.int64],
                                              shapes=[(3,), (3,)],
                                              shared_name=None, name=None)
        enqueue_op = input_queue.enqueue_many([image_paths_placeholder, labels_placeholder])

        images_and_labels = []
        for _ in range(NUM_PREPROCESS_THREADS):
            filenames, label = input_queue.dequeue()
            images = []
            for filename in tf.unstack(filenames):
                file_contents = tf.read_file(filename)
                image = tf.image.decode_image(file_contents, channels=3)

                image = tf.image.random_brightness(image, 0.2)
                image = tf.image.random_saturation(image, 0.6, 1.6)
                image = tf.image.random_contrast(image, 0.6, 1.4)
                image = tf.image.random_flip_left_right(image)

                # pylint: disable=no-member
                image.set_shape((INPUT_SIZE[0], INPUT_SIZE[1], 3))
                images.append(utils.tf_pre_process_image(image))
            images_and_labels.append([images, label])

        image_batch, labels_batch = tf.train.batch_join(
            images_and_labels, batch_size=batch_size_placeholder,
            shapes=[(INPUT_SIZE[0], INPUT_SIZE[1], 3), ()], enqueue_many=True,
            capacity=4 * NUM_PREPROCESS_THREADS * BATCH_SIZE,
            allow_smaller_final_batch=True)
        image_batch = tf.identity(image_batch, 'image_batch')
        labels_batch = tf.identity(labels_batch, 'label_batch')

        net = builder.input_and_train_node(input_layer, is_training) \
            .arch_type(MODEL) \
            .final_layer_type(FinalLayer.G) \
            .build()

        embeddings = tf.nn.l2_normalize(net.embedding, 1, 1e-10, name='embeddings')
        anchor, positive, negative = tf.unstack(tf.reshape(embeddings, [-1, 3, EMBEDDING_SIZE]), 3, 1)

        inference_loss = triplet_loss(anchor, positive, negative, ALPHA)
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
            step = 0
            try:
                for epoch_idx in range(EPOCH):
                    batch_idx = 0
                    while batch_idx < epoch_size:
                        # Select
                        image_paths, num_per_class = sample_people(dataset, PEOPLE_PER_BATCH, IMAGES_PER_PERSON)
                        # print('Running forward pass on sampled images: ', end='')
                        start_time = time.time()
                        nrof_examples = PEOPLE_PER_BATCH * IMAGES_PER_PERSON
                        labels_array = np.reshape(np.arange(nrof_examples), (-1, 3))
                        image_paths_array = np.reshape(np.expand_dims(np.array(image_paths), 1), (-1, 3))
                        sess.run(enqueue_op,
                                 {image_paths_placeholder: image_paths_array, labels_placeholder: labels_array})

                        emb_array = np.zeros((nrof_examples, EMBEDDING_SIZE))
                        nrof_batches = int(np.ceil(nrof_examples / BATCH_SIZE))
                        for i in range(nrof_batches):
                            batch_size = min(nrof_examples - i * BATCH_SIZE, BATCH_SIZE)
                            imgs, labs = sess.run([image_batch, labels_batch],
                                                  feed_dict={batch_size_placeholder: batch_size})

                            emb = sess.run([net.embedding], feed_dict={input_layer: imgs, is_training: True})
                            emb_array[labs, :] = emb

                        # print('%.3f' % (time.time() - start_time))
                        # print('Selecting suitable triplets for training')
                        triplets, nrof_random_negs, nrof_triplets = select_triplets(emb_array, num_per_class,
                                                                                    image_paths, PEOPLE_PER_BATCH,
                                                                                    ALPHA)
                        selection_time = time.time() - start_time
                        # print('(nrof_random_negs, nrof_triplets) = (%d, %d): time=%.3f seconds' %
                        #       (nrof_random_negs, nrof_triplets, selection_time))

                        # Training
                        nrof_batches = int(np.ceil(nrof_triplets * 3 / BATCH_SIZE))
                        triplet_paths = list(itertools.chain(*triplets))
                        labels_array = np.reshape(np.arange(len(triplet_paths)), (-1, 3))
                        triplet_paths_array = np.reshape(np.expand_dims(np.array(triplet_paths), 1), (-1, 3))
                        sess.run(enqueue_op,
                                 {image_paths_placeholder: triplet_paths_array, labels_placeholder: labels_array})
                        nrof_examples = len(triplet_paths)
                        i = 0
                        emb_array = np.zeros((nrof_examples, EMBEDDING_SIZE))
                        while i < nrof_batches:
                            start_time = time.time()
                            batch_size = min(nrof_examples - i * BATCH_SIZE, BATCH_SIZE)
                            imgs, labs = sess.run([image_batch, labels_batch],
                                                  feed_dict={batch_size_placeholder: batch_size})
                            feed_dict = {input_layer: imgs, is_training: True}
                            total_loss_val, inf_loss_val, wd_loss_val, _, step, emb = sess.run(
                                [total_loss, inference_loss, wd_loss, train_op, global_step, embeddings],
                                feed_dict=feed_dict)
                            emb_array[labs, :] = emb
                            duration = time.time() - start_time
                            batch_idx += 1
                            i += 1

                            # print training information
                            if step % SHOW_INFO_INTERVAL == 0:
                                label_name = [triplet_paths[l].split('/')[-2] for l in labs]
                                show_info(epoch_idx, batch_idx, epoch_size, step,
                                          total_loss_val, inf_loss_val, wd_loss_val, duration, emb, label_name)
                            # save summary
                            if step % SUMMARY_INTERVAL == 0:
                                save_summary(step, imgs, input_layer, sess, summary, summary_op)

                            # save ckpt files
                            if step % CKPT_INTERVAL == 0 and not have_best:
                                save_ckpt(step, epoch_idx, saver, sess)

                            # validate
                            if step % VALIDATE_INTERVAL == 0:
                                best_accuracy = validate(best_accuracy, step,
                                                         input_layer, net, saver, sess,
                                                         is_training, ver_dataset)
            except Exception as err:
                log('Exception, saving interrupt ckpt. err: {}'.format(err))
                filename = '{:s}_err_iter_{:d}.ckpt'.format(MODEL.name, step)
                filename = os.path.join(MODEL_OUT_PATH, filename)
                saver.save(sess, filename)
                raise err


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


def save_summary(step, images_train, input_layer, sess,
                 summary, summary_op):
    feed_summary_dict = {
        input_layer: images_train
    }
    summary_op_val = sess.run(summary_op, feed_dict=feed_summary_dict)
    summary.add_summary(summary_op_val, step)


def show_info(epoch_idx, batch_idx, epoch_size, step, total_loss_val, inf_loss_val, wd_loss_val,
              duration, emb, label_name):
    pos_dist = np.sum(np.square(emb[0] - emb[1]))
    neg_dist = np.sum(np.square(emb[0] - emb[2]))

    log('Epoch: [%d][%d/%d], step %d, total_loss: %.2f, inf_loss: %.2f, weight_loss: '
        '%.2f, Time %.3f, (A,P,N):(%s,%s,%s), (P,N):(%.3f,%.3f)' %
        (epoch_idx, batch_idx, epoch_size, step, total_loss_val, inf_loss_val, wd_loss_val,
         duration, label_name[0], label_name[1], label_name[2], pos_dist, neg_dist))


def sample_people(dataset, people_per_batch, images_per_person):
    nrof_images = people_per_batch * images_per_person

    # Sample classes from the dataset
    nrof_classes = len(dataset)
    class_indices = np.arange(nrof_classes)
    np.random.shuffle(class_indices)

    i = 0
    image_paths = []
    num_per_class = []
    sampled_class_indices = []
    # Sample images from these classes until we have enough
    while len(image_paths) < nrof_images:
        class_index = class_indices[i]
        nrof_images_in_class = len(dataset[class_index])
        image_indices = np.arange(nrof_images_in_class)
        np.random.shuffle(image_indices)
        nrof_images_from_class = min(nrof_images_in_class, images_per_person, nrof_images - len(image_paths))
        idx = image_indices[0:nrof_images_from_class]
        image_paths_for_class = [dataset[class_index].image_paths[j] for j in idx]
        sampled_class_indices += [class_index] * nrof_images_from_class
        image_paths += image_paths_for_class
        num_per_class.append(nrof_images_from_class)
        i += 1

    return image_paths, num_per_class


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

    dataset = [data for data in dataset if IMAGES_PER_PERSON <= len(data)]
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
