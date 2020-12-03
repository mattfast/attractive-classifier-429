import numpy as np
from tensorflow.keras import models
import pandas as pd
import cv2
from load_LFW_attributes import load_LFW_attributes, load_LFW_images
import os
from functools import cmp_to_key



def cmp(a, b):
    a = a[1]
    b = b[1]
    if a > b:
        return 1
    elif a == b:
        return 0
    else:
        return -1


"""img_files = []
for path, subdirs, files in os.walk('../fairface_val'):
    for name in files:
        if 'jpg' not in name:
            continue
        num = int(name.split('.')[0])
        img_files.append((os.path.join(path, name), num))
img_files = sorted(img_files, key=cmp_to_key(cmp))
img_files = [f[0] for f in img_files]
img_files = [cv2.imread(f) for f in img_files]
img_files = [cv2.resize(f, dsize=(250, 250)) for f in img_files]
img_files = [image / 255 for image in img_files]
img_files = np.array(img_files)"""

img_files = load_LFW_images('../lfw')
img_files = np.array(img_files)

m = models.load_model('../larger_model')
os.environ['KMP_DUPLICATE_LIB_OK']='True'
print('Finished Model Loading')

predictions = m.predict(img_files)
np.savetxt('../lfw_non_binary_trained_continuous_predictions.csv', predictions, delimiter=',')

