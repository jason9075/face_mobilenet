import cv2
import numpy as np
import tensorflow as tf

MODEL_PB = 'model_out/frozen_shape.pb'
OUTPUT_PATH = "events/"


def main():
    with tf.Session() as sess:
        saver = tf.train.import_meta_graph('model_out/InsightFace_iter_23000.ckpt.meta', clear_devices=True)
        saver.restore(sess, 'model_out/InsightFace_iter_23000.ckpt')

        # tf.summary.FileWriter(OUTPUT_PATH, graph=tf.get_default_graph())

        input_tensor = tf.get_default_graph().get_tensor_by_name("input_images:0")
        debug_tensor = tf.get_default_graph().get_tensor_by_name("mobilenet_v2/conv_1/bn/cond/Merge:0")

        image = cv2.imread('images/old_image/0_0a02b2.jpg')
        image = cv2.resize(image, (112, 112))
        image = image - 127.5
        image = image * 0.0078125
        image = np.expand_dims(image, axis=0)

        result = sess.run(debug_tensor, feed_dict={input_tensor: image})
        print(result)


if __name__ == '__main__':
    main()
