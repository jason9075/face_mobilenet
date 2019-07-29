import tfcoreml as tf_converter


def main():
    tf_converter.convert(tf_model_path='model_out/frozen_model.pb',
                         mlmodel_path='model_out/frozen_model.mlmodel',
                         output_feature_names=['g_type/embedding/Identity:0'],
                         input_name_shape_dict={'input_images:0': [1, 112, 112, 3]})


if __name__ == '__main__':
    main()
