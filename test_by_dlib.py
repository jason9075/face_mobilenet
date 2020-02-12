import face_recognition as fr
import numpy as np
import utils

SHAPE = (224, 224)
PLOT_ROC = True
LFW_PATH = 'tfrecord/lfw.bin'
VER_NAME = 'glint_tiny_verification.tfrecord'
VER_TYPE = 'euclidean'


def main():
    # estimator method
    def embedding_fn(img1, img2):
        h, w, _ = img1.shape
        img1 = img1 / 0.0078125 + 127.5
        img2 = img2 / 0.0078125 + 127.5
        img1 = img1.astype(np.uint8)
        img2 = img2.astype(np.uint8)
        result1 = fr.face_encodings(img1, [(0, w, h, 0)])[0]
        result2 = fr.face_encodings(img2, [(0, w, h, 0)])[0]
        vector_pair = utils.preprocessing.normalize([result1, result2])

        return vector_pair

    # gen_model()
    utils.test_tfrecord(VER_NAME, embedding_fn, SHAPE, is_plot=PLOT_ROC, ver_type=VER_TYPE)
    # utils.test_lfw(LFW_PATH, embedding_fn, SHAPE, is_plot=PLOT_ROC, ver_type=VER_TYPE)


if __name__ == '__main__':
    main()
