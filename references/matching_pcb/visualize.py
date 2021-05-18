import cv2 as cv
import numpy as np
from skimage.color import label2rgb
from skimage.morphology import diamond


def label_regions(image, mask, color=(255, 0, 0)):
    mask = (mask > 0).astype(np.uint8)

    # kernel = np.ones((3, 3), np.uint8)
    mask = cv.dilate(mask, diamond(2)) - mask

    area = np.stack([mask * c for c in color], axis=2)

    mask_inv = cv.bitwise_not(mask)
    img1 = cv.bitwise_and(area, area, mask=mask)
    img2 = cv.bitwise_and(image, image, mask=mask_inv)

    dst = cv.add(img1, img2)
    return dst


def label_regions2(image, mask, color=(255, 0, 0)):
    mask = (mask > 0).astype(np.uint8)

    area = np.stack([mask * c for c in color], axis=2)

    mask_inv = cv.bitwise_not(mask)
    img1 = cv.bitwise_and(area, area, mask=mask)
    img2 = cv.bitwise_and(image, image, mask=mask_inv)

    # cv.addWeighted(image, 0.7, dst, 0.3, 0)
    dst = cv.add(img1, img2)
    return dst


def label_to_rgb(label, image, color="red", alpha=0.3):
    return label2rgb(label, image, colors=[color], alpha=alpha,
                     bg_label=0, bg_color=None)
