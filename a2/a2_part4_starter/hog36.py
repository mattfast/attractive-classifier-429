"""
 Princeton University, COS 429, Fall 2020
"""
import numpy as np
from scipy.signal import convolve2d


def hog36(img, orientations, wrap180):
    """Computes a HoG descriptor for a 36x36 array, using 6x6 pixel cells,
       2x2 blocks, L2 normalization

    Args:
        img: a 36x36 image
        orientations: the number of gradient orientations to use
        wrap180: if true, the orientations cover 180 degrees 
        (i.e., there is no distinction between the gradient direction of 
        30 degrees or the polar opposite of 210 degrees). 

    Returns:
        descriptor: a HoG descriptor of length 100 * orientations
    """
    # You will need to modify this code to use input variable wrap180.
    # Currently this function always wraps orientations at 180 degrees.
    
    # Compute gradient using tiny centered-difference filter
    img = img.astype('float32')
    gx = convolve2d(img, [[1, 0, -1]], mode='same')
    gx[:, 0] = 0
    gx[:, -1] = 0
    gy = convolve2d(img, [[1], [0], [-1]], mode='same')
    gy[0, :] = 0
    gy[-1, :] = 0
    gmag = np.hypot(gx, gy)

    if wrap180:
        multiplier = orientations / np.pi
    else:
        multiplier = orientations / (2 * np.pi)

    gdir = np.arctan2(gy, gx) * multiplier % orientations

    # Bin by orientation
    cells = np.zeros([6, 6, orientations])
    filter = [1, 2, 3, 3, 2, 1]
    for i in range(orientations):
        # Select out by orientation, weight by gradient magnitude
        if i == orientations - 1:
            gdir_wrap = (gdir + 1) % orientations - 1
            this_orientation = gmag * np.maximum(1 - np.abs(gdir_wrap), 0)
        else:
            this_orientation = gmag * np.maximum(1 - np.abs(gdir - (i + 1)), 0)
        # Aggregate and downsample
        this_orientation = convolve2d(this_orientation, np.reshape(filter, [-1, 1]), mode='valid')
        this_orientation = convolve2d(this_orientation, np.reshape(filter, [1, -1]), mode='valid')
        cells[:, :, i] = this_orientation[::6, ::6]

    # Create the output vector
    descriptor = np.zeros(5 * 5 * 4 * orientations)
    for block_i in range(5):
        for block_j in range(5):
            block = cells[block_i:block_i+2, block_j:block_j+2, :]
            block_unrolled = block.flatten('F')
            norm_block_unrolled = np.linalg.norm(block_unrolled)
            if norm_block_unrolled > 0:
                block_unrolled = block_unrolled / np.linalg.norm(block_unrolled)
            else:
                block_unrolled = 1 / block_unrolled.shape[0]
            first = (5 * block_i + block_j) * 4 * orientations
            last = first + 4 * orientations
            descriptor[first:last] = block_unrolled
    return descriptor
