import numpy as np
import os

def load_MNIST_images(filename):
    """
    Args:
        filename: ubyte filename for the MNIST images

    Returns:
        images: [height] x [width] x 1 x [number of MNIST images] matrix containing the MNIST images.
                The images are of type float and scaled to [0, 1] for the convenience of training
    """

    intType = np.dtype('int32').newbyteorder('>')
    nMetaDataBytes = 4 * intType.itemsize

    data = np.fromfile(filename, dtype = 'ubyte')
    magicBytes, nImages, height, width = np.frombuffer(data[:nMetaDataBytes].tobytes(), intType)
    images = data[nMetaDataBytes:].astype(dtype = 'float32').reshape([nImages, height, width])

    images = np.expand_dims(np.transpose(images, [1, 2, 0]), 2)

    images = images / 255.0

    return images
