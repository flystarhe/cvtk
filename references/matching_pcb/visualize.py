import cv2 as cv
import numpy as np
from skimage.color import label2rgb
from skimage.morphology import diamond


def _color(mask, color):
    return np.stack([mask * c for c in color], axis=2)


def label_regions(image, masks, colors=(255, 0, 0), only_edge=True):
    if not isinstance(masks, list):
        masks = [masks]

    if not isinstance(colors, list):
        colors = [colors]

    masks = [(m > 0).astype(np.uint8) for m in masks]

    if only_edge:
        kernel = diamond(2)
        # kernel = np.ones((3, 3), np.uint8)
        masks = [cv.dilate(m, kernel) - m for m in masks]

    _mask = np.stack(masks, axis=2).sum(axis=2)
    _mask = (_mask > 0).astype(np.uint8)

    area = np.zeros_like(image, dtype=np.int32)
    for m, c in zip(masks, colors):
        area += _color(m, c)
    area = np.clip(area, 0, 255).astype(np.uint8)

    _mask = np.stack((_mask, _mask, _mask), axis=2)
    img1 = image * (1 - _mask)
    img2 = area * _mask

    return img1 + img2


def label_to_rgb(label, image, color="red", alpha=0.3):
    return label2rgb(label, image, colors=[color], alpha=alpha,
                     bg_label=0, bg_color=None)
