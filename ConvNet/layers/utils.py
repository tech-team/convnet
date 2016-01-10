import numpy as np


def im2col(matrix3d, window_size, stride, width, height):
    x_col = np.empty((window_size ** 2 * matrix3d.shape[2], width * height))

    i = 0
    for y in xrange(0, height):
        y_offset = y * stride
        for x in xrange(0, width):
            x_offset = x * stride
            region = matrix3d[x_offset:x_offset + window_size, y_offset:y_offset + window_size, :]
            x_col[:, i] = np.vstack([region[:, :, z].reshape(-1) for z in xrange(0, region.shape[2])]).reshape(-1)
            i += 1

    return x_col


def col2im(col, w, h, d):
    return col\
        .swapaxes(0, 1)\
        .reshape((w, h, d))\
        .swapaxes(0, 1)