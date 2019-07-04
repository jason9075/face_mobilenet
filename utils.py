import tensorflow as tf


def parse_function(example_proto):
    features = {'image_raw': tf.FixedLenFeature([], tf.string),
                'label': tf.FixedLenFeature([], tf.int64)}
    features = tf.parse_single_example(example_proto, features)
    # You can do more image distortion here for training data
    img = tf.image.decode_jpeg(features['image_raw'])
    img = tf.reshape(img, shape=(112, 112, 3))
    r, g, b = tf.split(img, num_or_size_splits=3, axis=-1)
    img = tf.concat([b, g, r], axis=-1)
    img = tf.cast(img, dtype=tf.float32)
    img = tf.subtract(img, 127.5)
    img = tf.multiply(img, 0.0078125)
    img = tf.image.random_flip_left_right(img)
    label = tf.cast(features['label'], tf.int64)
    return img, label


def test(data_set, sess, embedding_tensor, batch_size, label_shape=None, feed_dict=None, input_placeholder=None):
    print('testing verification..')
    data_list = data_set[0]
    issame_list = data_set[1]
    embeddings_list = []
    time_consumed = 0.0
    for i in range(len(data_list)):
        datas = data_list[i]
        embeddings = None
        feed_dict.setdefault(input_placeholder, None)
        for idx, data in enumerate(data_iter(datas, batch_size)):
            data_tmp = data.copy()    # fix issues #4
            data_tmp -= 127.5
            data_tmp *= 0.0078125
            feed_dict[input_placeholder] = data_tmp
            time0 = datetime.datetime.now()
            _embeddings = sess.run(embedding_tensor, feed_dict)
            time_now = datetime.datetime.now()
            diff = time_now - time0
            time_consumed += diff.total_seconds()
            if embeddings is None:
                embeddings = np.zeros((datas.shape[0], _embeddings.shape[1]))
            try:
                embeddings[idx*batch_size:min((idx+1)*batch_size, datas.shape[0]), ...] = _embeddings
            except ValueError:
                print('idx*batch_size value is %d min((idx+1)*batch_size, datas.shape[0]) %d, batch_size %d, data.shape[0] %d' %
                      (idx*batch_size, min((idx+1)*batch_size, datas.shape[0]), batch_size, datas.shape[0]))
                print('embedding shape is ', _embeddings.shape)
        embeddings_list.append(embeddings)

    _xnorm = 0.0
    _xnorm_cnt = 0
    for embed in embeddings_list:
        for i in range(embed.shape[0]):
            _em = embed[i]
            _norm = np.linalg.norm(_em)
            # print(_em.shape, _norm)
            _xnorm += _norm
            _xnorm_cnt += 1
    _xnorm /= _xnorm_cnt

    acc1 = 0.0
    std1 = 0.0
    embeddings = embeddings_list[0] + embeddings_list[1]
    embeddings = sklearn.preprocessing.normalize(embeddings)
    print(embeddings.shape)
    print('infer time', time_consumed)
    _, _, accuracy, val, val_std, far = evaluate(embeddings, issame_list, nrof_folds=10)
    acc2, std2 = np.mean(accuracy), np.std(accuracy)
    return acc1, std1, acc2, std2, _xnorm, embeddings_list


def ver_test(ver_list, ver_name_list, nbatch, sess, embedding_tensor, batch_size, feed_dict, input_placeholder):
    results = []
    for i in range(len(ver_list)):
        acc1, std1, acc2, std2, xnorm, embeddings_list = test(data_set=ver_list[i], sess=sess, embedding_tensor=embedding_tensor,
                                                              batch_size=batch_size, feed_dict=feed_dict,
                                                              input_placeholder=input_placeholder)
        print('[%s][%d]XNorm: %f' % (ver_name_list[i], nbatch, xnorm))
        print('[%s][%d]Accuracy-Flip: %1.5f+-%1.5f' % (ver_name_list[i], nbatch, acc2, std2))
        results.append(acc2)
    return results