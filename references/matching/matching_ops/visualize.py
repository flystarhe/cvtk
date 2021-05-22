import cv2 as cv
import numpy as np
from skimage.color import label2rgb
from skimage.morphology import diamond

from matching_ops.image import to_rgb


def _color(mask, color):
    return np.stack([mask * c for c in color], axis=2)


def draw_bbox(image, bbox, color=(255, 0, 0)):
    x1, y1, w, h = list(map(int, bbox))
    pt1, pt2 = (x1, y1), (x1 + w, y1 + h)
    return cv.rectangle(image, pt1, pt2, color)


def show_image(image, color="RGB"):
    from IPython.display import display
    from PIL import Image as Image2

    rgb = to_rgb(image, color=color)
    display(Image2.fromarray(rgb, "RGB"))


def show_images(images, cols=2, color="RGB"):
    from IPython.display import display
    from PIL import Image as Image2

    images = [to_rgb(img, color=color) for img in images]

    h_ = min([img.shape[0] for img in images])
    w_ = min([img.shape[1] for img in images])
    images = [img[:h_, :w_] for img in images]

    for i in range(0, len(images), cols):
        batch_rgb = np.concatenate(images[i:i + cols], axis=1)
        display(Image2.fromarray(batch_rgb, "RGB"))


def label_regions(image, masks, colors=(255, 0, 0), only_edge=True):
    if not isinstance(masks, list):
        masks = [masks]

    if not isinstance(colors, list):
        colors = [colors]

    masks = [(m > 0).astype(np.uint8) for m in masks]

    if only_edge:
        kernel = diamond(2)
        # kernel = np.ones((3, 3), np.uint8)
        masks = [cv.dilate(m, kernel) - m for m in masks]

    _mask = np.stack(masks, axis=2).sum(axis=2)
    _mask = (_mask > 0).astype(np.uint8)

    area = np.zeros_like(image, dtype=np.int32)
    for m, c in zip(masks, colors):
        area += _color(m, c)
    area = np.clip(area, 0, 255).astype(np.uint8)

    _mask = np.stack((_mask, _mask, _mask), axis=2)
    img1 = image * (1 - _mask)
    img2 = area * _mask

    return img1 + img2


def label_to_rgb(label, image, color="red", alpha=0.3):
    return label2rgb(label, image, colors=[color], alpha=alpha,
                     bg_label=0, bg_color=None)
