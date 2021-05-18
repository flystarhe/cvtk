"""Align with binary or gray"""
import cv2 as cv
import numpy as np


TM_MINIMUM_METHODS = (cv.TM_SQDIFF, cv.TM_SQDIFF_NORMED)


def binary_matching(image, templ, method=cv.TM_CCOEFF_NORMED):
    if isinstance(method, str):
        method = getattr(cv, method)
    res = cv.matchTemplate(image, templ, method)

    min_val, max_val, min_loc, max_loc = cv.minMaxLoc(res)

    if method in TM_MINIMUM_METHODS:
        max_val = 1.0 - min_val
        max_loc = min_loc

    return max_val, max_loc


def _resize(image, f=1.0):
    im_h, im_w = image.shape[:2]
    dsize = (int(im_w * f + 0.5), int(im_h * f + 0.5))
    return cv.resize(image, dsize, interpolation=cv.INTER_LINEAR)


def _blocks(size, f=0.7):
    im_h, im_w = size[:2]
    h_, w_ = int(im_h * f), int(im_w * f)
    return [(0, h_, 0, w_), (0, h_, im_w - w_, im_w),
            (im_h - h_, im_h, 0, w_), (im_h - h_, im_h, im_w - w_, im_w)]


def search_blocks(image, templ, method, **kw):
    factor = kw.get("factor", 0.7)
    threshold = kw.get("threshold", 0.9)

    pts = []
    for y1, y2, x1, x2 in _blocks(templ.shape[:2], f=factor):
        max_val, max_loc = binary_matching(image, templ[y1:y2, x1:x2], method)
        pts.append((max_loc[0] - x1, max_loc[1] - y1, max_val))
        if max_val >= threshold:
            break
    return pts


def search_center(image, templ, method, **kw):
    factor = (1.0 - kw.get("factor", 0.7)) * 0.5

    im_h, im_w = templ.shape[:2]

    y1, x1 = int(im_h * factor), int(im_w * factor)
    y2, x2 = im_h - y1, im_w - x1

    max_val, max_loc = binary_matching(image, templ[y1:y2, x1:x2], method)
    return [(max_loc[0] - x1, max_loc[1] - y1, max_val)]


def aligned_pairs(image, templ, mode="center", factor=0.7, **kw):
    method = kw.pop("method", cv.TM_CCOEFF_NORMED)
    kw["factor"] = factor

    if mode == "center":
        pts = search_center(image, templ, method, **kw)
    elif mode == "blocks":
        pts = search_blocks(image, templ, method, **kw)
    else:
        max_val, max_loc = binary_matching(image, templ, method)
        pts = [(max_loc[0], max_loc[1], max_val)]

    x0, y0, score = max(pts, key=lambda x: x[-1])

    x1, p1 = (x0, 0) if x0 >= 0 else (0, -x0)
    y1, q1 = (y0, 0) if y0 >= 0 else (0, -y0)

    w = min(image.shape[1] - x1, templ.shape[1] - p1)
    h = min(image.shape[0] - y1, templ.shape[0] - q1)

    return (x1, y1, w, h), (p1, q1, w, h), score


def ez_pairs(image, templ, mode="center", factor=0.7, **kw):
    # kernel = disk(radius=kw.pop("radius", 1))
    kernel = np.ones((3, 3), dtype=np.uint8)
    threshold = kw.get("threshold", 0.9)

    data = []
    for _ in range(3):
        bbox1, bbox2, score = aligned_pairs(image, templ, mode, factor, **kw)
        data.append((bbox1, bbox2, score))
        if score >= threshold:
            break

        templ = cv.erode(templ, kernel)

    return max(data, key=lambda x: x[-1])
