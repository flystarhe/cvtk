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


def draw_bbox(image, bbox, color=(255, 0, 0)):
    x1, y1, w, h = list(map(int, bbox))
    pt1, pt2 = (x1, y1), (x1 + w, y1 + h)
    return cv.rectangle(image, pt1, pt2, color)


def show_image(image, color="RGB"):
    from PIL import Image as Image2
    from IPython.display import display

    rgb = to_rgb(image, color=color)
    display(Image2.fromarray(rgb, "RGB"))


def show_images(images, cols=2, color="RGB"):
    from PIL import Image as Image2
    from IPython.display import display

    images = [to_rgb(img, color=color) for img in images]

    h_ = min([img.shape[0] for img in images])
    w_ = min([img.shape[1] for img in images])
    images = [img[:h_, :w_] for img in images]

    for i in range(0, len(images), cols):
        batch_rgb = np.concatenate(images[i:i + cols], axis=1)
        display(Image2.fromarray(batch_rgb, "RGB"))
