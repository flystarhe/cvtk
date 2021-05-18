import cv2 as cv
import numpy as np


def threshold_real(image, color="BGR", blurring="gauss", ksize=7):
    if isinstance(image, str):
        image = cv.imread(image, 0)

    assert isinstance(image, np.ndarray)

    if image.ndim == 3:
        code = getattr(cv, f"COLOR_{color.upper()}2GRAY")
        image = cv.cvtColor(image, code)

    if blurring == "gauss":
        ksize = (ksize, ksize)
        image = cv.GaussianBlur(image, ksize, 0)
    elif blurring == "median":
        image = cv.medianBlur(image, ksize)

    _, dst = cv.threshold(image, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)
    return dst
