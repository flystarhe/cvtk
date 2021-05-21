from pathlib import Path

import cv2 as cv
import hiplot as hip
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt

IMG_EXTENSIONS = set([".jpg", ".jpeg", ".png", ".bmp"])


def _hist(npimg, n=256):
    return np.bincount(npimg.ravel(), minlength=n)


def _hist2d(nparr, a, b, n=256):
    return pd.DataFrame({f"c{i}": nparr[..., i].ravel() for i in (a, b)})


def _hist2d_matrix(nparr, a, b, n=256):
    npimg = n * nparr[..., a] + nparr[..., b]
    matrix = np.bincount(npimg.ravel(), minlength=n**2).reshape(n, n)

    h, w = np.nonzero(matrix >= matrix.max() * 0.05)
    x1, x2 = w.min(), w.max() + 1
    y1, y2 = h.min(), h.max() + 1

    matrix = matrix[y1:y2, x1:x2]
    return matrix, x1, y1, x2, y2


def _hist2d_dataframe(nparr, a, b, n=256, labels=None):
    matrix, x1, y1, x2, y2 = _hist2d_matrix(nparr, a, b, n)

    if labels is None:
        labels = [f"C{i}" for i in range(a + b + 1)]

    rows = pd.Index(np.arange(y1, y2), name=labels[a])
    cols = pd.Index(np.arange(x1, x2), name=labels[b])
    return pd.DataFrame(matrix, index=rows, columns=cols)


def color_hist(img_path, start=0, limit=1, mode="BGR"):
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
        nparr = cv.imread(str(img_path), 1)

        if mode == "BGR":
            labels = ["B", "G", "R"]
        elif mode == "HSV":
            labels = ["H", "S", "V"]
            nparr = cv.cvtColor(nparr, cv.COLOR_BGR2HSV)
        elif mode == "YCrCb":
            labels = ["Y", "Cr", "Cb"]
            nparr = cv.cvtColor(nparr, cv.COLOR_BGR2YCrCb)
        else:
            raise Exception(f"{mode} not supported")

        for i in (0, 1, 2):
            hist = _hist(nparr[..., i])
            axs[i].plot(hist / hist.max(), label="count")
            hist = np.cumsum(hist)
            axs[i].plot(hist / hist.max(), label="cumsum")
            axs[i].set_title(f"{labels[i]} of {mode}")
            axs[i].set_xlim(0, 256)
            axs[i].legend()


def color_hist2d(img_path, start=0, limit=1, mode="BGR"):
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

    img_path = img_list[start]
    nparr = cv.imread(str(img_path), 1)

    if mode == "BGR":
        labels = ["B", "G", "R"]
    elif mode == "HSV":
        labels = ["H", "S", "V"]
        nparr = cv.cvtColor(nparr, cv.COLOR_BGR2HSV)
    elif mode == "YCrCb":
        labels = ["Y", "Cr", "Cb"]
        nparr = cv.cvtColor(nparr, cv.COLOR_BGR2YCrCb)
    else:
        raise Exception(f"{mode} not supported")

    for a, b in [(0, 1), (0, 2), (1, 2)]:
        df = _hist2d_dataframe(nparr, a, b, 256, labels)
        fig, ax = plt.subplots(figsize=(9, 6))
        sns.heatmap(df, cmap="jet", ax=ax)


def color_hiplot(img_path, start=0, limit=1, mode="BGR", max_rows=10000):
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

    img_path = img_list[start]
    nparr = cv.imread(str(img_path), 1)

    if mode == "BGR":
        labels = ["B", "G", "R"]
    elif mode == "HSV":
        labels = ["H", "S", "V"]
        nparr = cv.cvtColor(nparr, cv.COLOR_BGR2HSV)
    elif mode == "YCrCb":
        labels = ["Y", "Cr", "Cb"]
        nparr = cv.cvtColor(nparr, cv.COLOR_BGR2YCrCb)
    else:
        raise Exception(f"{mode} not supported")

    df = pd.DataFrame({label: nparr[..., i].ravel()
                       for i, label in enumerate(labels)})

    if df.shape[0] > max_rows:
        df = df.sample(n=max_rows, random_state=100)

    hip.Experiment.from_dataframe(df).display()
