import cv2 as cv
import numpy as np


def to_rgb(image, color="BGR", max_val=None):
    if isinstance(image, str):
        image = cv.imread(image, 1)
        color = "BGR"

    assert isinstance(image, np.ndarray)

    if image.dtype != np.uint8:
        if max_val is None:
            min_val = image.min()
            max_val = image.max()
            scale = max_val - min_val
            if scale > 1e-3:
                image = (image - min_val) / scale
        else:
            image = image / max_val

        image = np.clip(image * 255, 0, 255)
        image = image.astype(np.uint8)

    if image.ndim == 2:
        image = cv.cvtColor(image, cv.COLOR_GRAY2RGB)
        color = "RGB"

    if color != "RGB":
        code = getattr(cv, f"COLOR_{color.upper()}2RGB")
        image = cv.cvtColor(image, code)

    return image


def crop_image(image, bbox):
    x1, y1, w, h = list(map(int, bbox))
    return image[y1:y1 + h, x1:x1 + w]
