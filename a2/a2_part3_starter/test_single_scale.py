"""
 Princeton University, COS 429, Fall 2020
"""
import os
import cv2
import numpy as np
from glob import glob
import matplotlib.pyplot as plt 
from find_faces_single_scale import find_faces_single_scale


def test_single_scale(stride, thresh):
    """Test face detection on all single-scale images

    Args:
        stride: how far to move between locations at which the detector is run
        thresh: probability threshold for calling a detection a face
    """
    saved = np.load('face_classifier.npz')
    params, orientations, wrap180 = saved['params'], saved['orientations'], saved['wrap180']
    single_scale_scenes_dir = 'face_data/single_scale_scenes'
    scene_filenames = sorted(glob(os.path.join(single_scale_scenes_dir, '*.jpg')))

    for filename in scene_filenames:
        print('Detecting faces in', filename)

        img = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)
        outimg, probmap = find_faces_single_scale(img, stride, thresh, params, orientations, wrap180)

        plt.figure()
        plt.title(filename)
        plt.imshow(outimg, cmap='gray')
        plt.show(block=False)
        input('Press enter to continue')


if __name__ == '__main__':
    test_single_scale(3, 0.95)
