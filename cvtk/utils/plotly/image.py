from pathlib import Path

import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt

IMG_EXTENSIONS = set([".jpg", ".jpeg", ".png", ".bmp"])


def _hist(npimg):
    hist = np.bincount(npimg.ravel(), minlength=256)
    v_sum, h_sum = np.sum(npimg, axis=1), np.sum(npimg, axis=0)
    return hist, v_sum, h_sum


def imhist(img_dir, start=0, limit=1):
    # jupyter: %matplotlib inline
    img_list = sorted(Path(img_dir).glob("**/*"))
    img_list = [x for x in img_list if x.suffix in IMG_EXTENSIONS]

    fig, axs = plt.subplots(nrows=3, figsize=(8, 12))
    for img_path in img_list[start: start + limit]:
        npimg = cv.imread(str(img_path), 0)
        hist, v_sum, h_sum = _hist(npimg)

        ax = axs[0]
        ax.plot(hist)
        ax.set_title("Image gray value")
        ax.set_xlim(0, 256)

        ax = axs[1]
        ax.plot(v_sum)
        ax.set_title("Vertical projection")

        ax = axs[2]
        ax.plot(h_sum)
        ax.set_title("Horizontal projection")
