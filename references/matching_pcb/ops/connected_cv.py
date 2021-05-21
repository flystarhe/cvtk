import cv2 as cv
import numpy as np


def connection():
    pass


def neighbors(stats1, stats2, max_val=1e5):
    """stats from cv.connectedComponentsWithStats."""
    pts1 = np.concatenate(
        (stats1[:, :2], stats1[:, :2] + stats1[:, 2:4]), axis=0)
    pts2 = np.concatenate(
        (stats2[:, :2], stats2[:, :2] + stats2[:, 2:4]), axis=0)

    dist = np.abs(pts1[:, None] - pts2).sum(axis=2)
    eye = np.eye(dist.shape[0], dtype=dist.dtype)
    R = (dist + eye * max_val).argmin(axis=1)
    return R.reshape((2, -1)).T


def neighbors2(index, labels1, labels2, simple=False):
    """labels from cv.connectedComponentsWithStats."""
    mask = (labels1 == index).astype(np.uint8)

    prop = None
    if not simple:
        pass

    labels = cv.bitwise_and(labels2, labels2, mask=mask)
    n_neighbors = np.unique(labels).tolist()[1:]
    return prop, n_neighbors
