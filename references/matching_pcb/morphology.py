import cv2 as cv
import numpy as np
from skimage.morphology import disk


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


def revise_templ(templ, image, **kw):
    radius = kw.pop("radius", 10)
    kernel = disk(radius=radius)

    shape = (radius * 2 + 1, radius * 2 + 1)
    kernel_ext = np.ones(shape, np.uint8)

    erosion = binary_erode(templ, kernel)
    region = templ - erosion

    bg = cv.bitwise_not(image)
    bg_region = cv.bitwise_and(region, bg)

    templ = templ - bg_region
    templ = binary_closing(templ, kernel_ext)

    return templ


def revise_templ2(templ, image,  **kw):
    radius = kw.pop("radius", 10)
    kernel = disk(radius=radius)

    shape = (radius * 2 + 1, radius * 2 + 1)
    kernel_ext = np.ones(shape, np.uint8)

    erosion = binary_erode(templ, kernel)
    region = templ - erosion

    bg = cv.bitwise_not(image)
    bg_region = cv.bitwise_and(region, bg)

    bg_region_closing = binary_closing(bg_region, kernel_ext)
    reduce = cv.bitwise_and(region, bg_region_closing)
    templ = templ - reduce

    region = binary_dilate(templ, kernel)

    fg_region = cv.bitwise_and(region, image)
    bg_region = cv.bitwise_not(fg_region)

    bg_region_closing = binary_closing(bg_region, kernel_ext)
    reduce = cv.bitwise_and(region, bg_region_closing)
    templ = region - reduce

    return templ
