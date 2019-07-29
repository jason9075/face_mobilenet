# please use command to run script
import tensorflow as tf


def main():
    pb_path = 'model_out/frozen_model.pb'
    input_node = 'input_images'
    output_node = 'gdc/embedding/Identity'

    converter = tf.lite.TFLiteConverter.from_frozen_graph(pb_path, [input_node], [output_node],
                                                          input_shapes={input_node: [1, 112, 112, 3]})
    tflite_model = converter.convert()

    with open('model_out/frozen_model.tflite', "wb") as f:
        f.write(tflite_model)


if __name__ == '__main__':
    main()
