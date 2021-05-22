import cv2 as cv
import numpy as np


def binary_closing(image, kernel, iterations=1):
    if isinstance(kernel, int):
        kernel = np.ones((kernel, kernel), np.uint8)
    return cv.morphologyEx(image, cv.MORPH_CLOSE,
                           kernel, iterations=iterations)


def binary_opening(image, kernel, iterations=1):
    if isinstance(kernel, int):
        kernel = np.ones((kernel, kernel), np.uint8)
    return cv.morphologyEx(image, cv.MORPH_OPEN,
                           kernel, iterations=iterations)


def binary_dilate(image, kernel, iterations=1):
    if isinstance(kernel, int):
        kernel = np.ones((kernel, kernel), np.uint8)
    return cv.dilate(image, kernel, iterations=iterations)


def binary_erode(image, kernel, iterations=1):
    if isinstance(kernel, int):
        kernel = np.ones((kernel, kernel), np.uint8)
    return cv.erode(image, kernel, iterations=iterations)
