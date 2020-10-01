"""
 Princeton University, COS 429, Fall 2020
"""
import numpy as np
from hog36 import hog36
from logistic_prob import logistic_prob


def find_faces_single_scale(img, stride, thresh, params, orientations, wrap180):
    """Find 36x36 faces in an image

    Args:
        img: an image
        stride: how far to move between locations at which the detector is run
        thresh: probability threshold for calling a detection a face
        params: trained face classifier parameters
        orientations: the number of HoG gradient orientations to use
        wrap180: if true, the HoG orientations cover 180 degrees, else 360

    Returns:
        outimg: copy of img with face locations marked
        probmap: probability map of face detections
    """
    windowsize = 36
    if stride > windowsize:
        stride = windowsize

    hog_descriptor_size = 100 * orientations
    height, width = img.shape
    probmap = np.zeros([height, width])
    outimg = np.array(img)

    for row in range(0, height - windowsize, stride):
        for col in range(0, width - windowsize, stride):
            crop = img[row:row+windowsize, col:col+windowsize]
            descriptors = np.zeros([1, hog_descriptor_size + 1])

            face_descriptor = hog36(crop, orientations, wrap180)
            descriptors[0, 0] = 1
            descriptors[0, 1:] = face_descriptor

            probability = logistic_prob(descriptors, params)[0]

            # Mark detection probability in probmap
            win_i = row + int((windowsize - stride) / 2)
            win_j = col + int((windowsize - stride) / 2)
            probmap[win_i:win_i+stride, win_j:win_j+stride] = probability

            # If probability of a face is below thresh, continue
            if probability < thresh:
                continue

            # Mark the face in outimg
            outimg[row, col:col+windowsize] = 255
            outimg[row+windowsize-1, col:col+windowsize] = 255
            outimg[row:row+windowsize, col] = 255
            outimg[row:row+windowsize, col+windowsize-1] = 255

    return outimg, probmap
