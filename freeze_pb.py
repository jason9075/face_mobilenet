import cv2

import tensorflow as tf
from tensorflow.python.platform import gfile
import numpy as np

import utils

CKPT_NAME = 'InsightFace_iter_198000.ckpt'


def main():
    img = cv2.imread('images/image_db/andy/gen_3791a1_13.jpg')

    img = utils.pre_process_image(img)

    with tf.Session() as sess:
        saver = tf.train.import_meta_graph(
            f'model_out/{CKPT_NAME}.meta', clear_devices=True)
        saver.restore(sess, f"model_out/{CKPT_NAME}")

        additional_nodes = ['gdc/embedding/Identity']
        frozen_graph = freeze_session(sess, output_names=additional_nodes)

        tf.summary.FileWriter("output_models/", graph=frozen_graph)
        tf.io.write_graph(frozen_graph, "output_models/", "frozen_model.pb", as_text=True)

    # with tf.Session() as sess:
    #     graph_def = tf.GraphDef()
    #     with gfile.FastGFile('output_models/frozen_model.pb', 'rb') as f:
    #         graph_def.ParseFromString(f.read())
    #     tf.import_graph_def(graph_def, name='')
    #     input_tensor = tf.get_default_graph().get_tensor_by_name(
    #         "input_images:0")
    #     trainable = tf.get_default_graph().get_tensor_by_name("trainable_bn:0")
    #     embedding_tensor = tf.get_default_graph().get_tensor_by_name(
    #         "gdc/embedding/Identity:0")
    #     tf.summary.FileWriter("output_models/", graph=tf.get_default_graph())
    #     result = sess.run(embedding_tensor, feed_dict={input_tensor: np.expand_dims(img, axis=0), trainable: False})
    #     print(result)


def freeze_session(session, keep_var_names=None, output_names=None,blacklist=None, clear_devices=True):
    graph = session.graph
    with graph.as_default():
        # freeze_var_names = list(set(v.op.name for v in tf.global_variables()).difference(keep_var_names or []))
        output_names = output_names or []
        # output_names += [v.op.name for v in tf.global_variables()]
        input_graph_def = graph.as_graph_def()
        if clear_devices:
            for node in input_graph_def.node:
                node.device = ""
        frozen_graph = tf.graph_util.convert_variables_to_constants(
            session, input_graph_def, output_names)
        return frozen_graph


if __name__ == '__main__':
    main()
