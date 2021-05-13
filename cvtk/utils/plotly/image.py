from pathlib import Path

import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt

IMG_EXTENSIONS = set([".jpg", ".jpeg", ".png", ".bmp"])


def _hist(npimg):
    hist = np.bincount(npimg.ravel(), minlength=256)
    v_sum, h_sum = np.sum(npimg, axis=1), np.sum(npimg, axis=0)
    return hist, v_sum, h_sum


def image_hist(img_path, start=0, limit=1):
    # jupyter: %matplotlib inline
    img_path = Path(img_path)

    if img_path.is_file():
        img_list = [img_path]
    elif img_path.is_dir():
        img_list = sorted(img_path.glob("**/*"))
        img_list = [x for x in img_list if x.suffix in IMG_EXTENSIONS]
    else:
        raise TypeError(f"{img_path} is not a file or dir")

    assert limit >= 1 and len(img_list) > start

    fig, axs = plt.subplots(nrows=3, figsize=(8, 12))
    for img_path in img_list[start: start + limit]:
        npimg = cv.imread(str(img_path), 0)
        hist, v_sum, h_sum = _hist(npimg)

        ax = axs[0]
        ax.plot(hist / (hist.max() + 1))
        ax.set_title("Image grayscale value")
        ax.set_xlim(0, 256)

        ax = axs[1]
        ax.plot(v_sum / (v_sum.max() + 1))
        ax.set_title("Vertical projection")

        ax = axs[2]
        ax.plot(h_sum / (h_sum.max() + 1))
        ax.set_title("Horizontal projection")
