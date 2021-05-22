import cv2 as cv
import numpy as np
from skimage.morphology import disk

from matching_pcb.connected_cv import neighbors2
from matching_pcb.morphology import binary_closing, binary_dilate, binary_erode


def expand_region(templ, image, kernel):
    pass


def reduce_region(templ, image, kernel, factor=0.5, small_area=50, **kw):
    erosion = binary_erode(templ, kernel)
    region = templ - erosion

    fg = cv.bitwise_and(region, image)
    bg = cv.bitwise_xor(region, fg)

    retval_fg, labels_fg, stats_fg, _ = cv.connectedComponentsWithStats(
        fg, connectivity=8, ltype=cv.CV_16U)
    retval_bg, labels_bg, stats_bg, _ = cv.connectedComponentsWithStats(
        bg, connectivity=8, ltype=cv.CV_16U)

    _labels_fg = labels_fg.copy()
    _labels_bg = labels_bg.copy()

    kernel_ = np.ones((3, 3), np.uint8)
    labels_fg = cv.dilate(labels_fg, kernel_)
    labels_bg = cv.dilate(labels_bg, kernel_)

    stats_fg = stats_fg.tolist()
    stats_bg = stats_bg.tolist()

    cache = np.zeros_like(templ)
    for index in range(1, retval_fg):
        _, _, _w, _h, _area = stats_fg[index]
        if _area >= small_area:
            continue

        prop, n_neighbors = neighbors2(index, labels_fg, labels_bg)
        area = sum([stats_bg[label][4] for label in n_neighbors])
        if area * factor > _area:
            patch = (_labels_fg == index).astype(np.uint8)
            cache = cache + patch
    bg = bg + cache

    cache = np.zeros_like(templ)
    for index in range(1, retval_bg):
        _, _, _w, _h, _area = stats_bg[index]
        if _area >= small_area:
            continue

        prop, n_neighbors = neighbors2(index, labels_bg, labels_fg)
        area = sum([stats_fg[label][4] for label in n_neighbors])
        if area * factor > _area:
            patch = (_labels_bg == index).astype(np.uint8)
            cache = cache + patch
    bg = bg - cache

    return templ - bg


def revise_templ(templ, image, radius=2, iters=1, **kw):
    kernel = disk(radius=radius)

    for _ in range(iters):
        templ = reduce_region(templ, image, kernel, **kw)

    return templ


def revise_templ2(templ, image, **kw):
    radius = kw.pop("radius", 10)
    kernel = disk(radius=radius)

    ksize = radius * 2 + 1
    kernel_ext = np.ones((ksize, ksize), np.uint8)

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
