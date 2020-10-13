import cv2
import numpy as np
import matplotlib.pyplot as plt
from find_faces import find_faces


img = cv2.imread('face_data/testing_scenes/people.jpg', cv2.IMREAD_GRAYSCALE)
saved = np.load('face_classifier.npz')
params, orientations, wrap180 = saved['params'], saved['orientations'], saved['wrap180']
outimg, probmap = find_faces(img, 3, 0.98, params, orientations, wrap180)

plt.figure(1)
plt.title('outimg')
plt.imshow(outimg, cmap='gray')
plt.show()
plt.figure(2)
plt.title('probmap')
plt.imshow(probmap, cmap='gray')
plt.show()