import glob
import os

import cv2
import numpy as np
from cv_core.detector.face_inference import FaceLandmarksDetector
from skimage import transform as trans

# full width/height is 112
SIZE = 112
left_profile_landmarks = np.array([
    [51.642, 50.115],
    [57.617, 49.990],
    [35.740, 69.007],
    [51.157, 89.050],
    [57.025, 89.702]], dtype=np.float32)

left_landmarks = np.array([
    [45.031, 50.118],
    [65.568, 50.872],
    [39.677, 68.111],
    [45.177, 86.190],
    [64.246, 86.758]], dtype=np.float32)

front_landmarks = np.array([
    [39.730, 51.138],
    [72.270, 51.138],
    [56.000, 68.493],
    [42.463, 87.010],
    [69.537, 87.010]], dtype=np.float32)

right_landmarks = np.array([
    [46.845, 50.872],
    [67.382, 50.118],
    [72.737, 68.111],
    [48.167, 86.758],
    [67.236, 86.190]], dtype=np.float32)

right_profile_landmarks = np.array([
    [54.796, 49.990],
    [60.771, 50.115],
    [76.673, 69.007],
    [55.388, 89.702],
    [61.257, 89.050]], dtype=np.float32)

landmark_src = np.array([left_profile_landmarks, left_landmarks,
                         front_landmarks, right_landmarks, right_profile_landmarks])


def estimate_norm(lmk):
    assert lmk.shape == (5, 2)
    tform = trans.SimilarityTransform()
    lmk_tran = np.insert(lmk, 2, values=np.ones(5), axis=1)
    min_M = []
    min_index = []
    min_error = float('inf')
    for i in range(5):
        tform.estimate(lmk, landmark_src[i])
        M = tform.params[0:2, :]
        results = np.dot(M, lmk_tran.T)
        results = results.T
        error = np.sum(np.sqrt(np.sum((results - landmark_src[i]) ** 2, axis=1)))
        if error < min_error:
            min_error = error
            min_M = M
            min_index = i
    return min_M, min_index


def align(img, landmark, image_size):
    M, pose_index = estimate_norm(landmark)
    warped = cv2.warpAffine(img, M, (image_size, image_size), borderValue=0.0)
    return warped


def main():
    landmarks_detector = FaceLandmarksDetector()
    # img = cv2.imread('images/0001_01.png')
    # img = cv2.resize(img, (SIZE, SIZE))
    # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    for name in os.listdir('images/public_face_1036_224/'):
        if name == '.DS_Store':
            continue
        os.mkdir(f'images/output/{name}')
        for img_path in glob.glob(f'images/public_face_1036_224/{name}/*.jpg'):
            img_name = img_path.split('/')[-1]
            img = cv2.imread(img_path)
            img = cv2.resize(img, (SIZE, SIZE))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            landmarks = landmarks_detector.predict(img)
            left_eye_ball = ((landmarks['left_eye'][0][0] + landmarks['left_eye'][3][0]) / 2,
                             (landmarks['left_eye'][0][1] + landmarks['left_eye'][3][1]) / 2,)
            right_eye_ball = ((landmarks['right_eye'][0][0] + landmarks['right_eye'][3][0]) / 2,
                              (landmarks['right_eye'][0][1] + landmarks['right_eye'][3][1]) / 2)
            landmark = np.asarray([left_eye_ball, right_eye_ball,
                                   landmarks['nose_bridge'][3], landmarks['top_lip'][0], landmarks['top_lip'][6]])

            img = align(img, landmark, SIZE)

            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            cv2.imwrite(f'images/output/{name}/{img_name}', img)


if __name__ == '__main__':
    main()
