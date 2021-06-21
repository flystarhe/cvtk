import os

import cv2 as cv
import numpy as np
import torch
from skimage.color import label2rgb, rgb_colors
from torch.nn import functional as F

DEFAULT_COLORS = ("red", "blue", "yellow", "magenta", "green",
                  "indigo", "darkorange", "cyan", "pink", "yellowgreen")


def draw_legend(npimg, names=None, colors=None):
    npimg = np.ascontiguousarray(npimg)
    if npimg.dtype != np.uint8:
        npimg = np.clip(npimg * 255, 0, 255)
        npimg = npimg.astype(np.uint8)

    if names is None or colors is None:
        return npimg

    color_dict = {k: v for k, v in rgb_colors.__dict__.items()
                  if isinstance(v, tuple)}

    colors = [np.array(color_dict[c][:3]) for c in colors]
    colors = [np.clip(c * 255, 0, 255).astype(np.uint8).tolist()
              for c in colors]

    for i, (color, name) in enumerate(zip(colors, names)):
        x1, y1, x2, y2 = 20, 20 + i * 40, 50, 50 + i * 40
        cv.rectangle(npimg, (x1, y1), (x2, y2), color, thickness=2)
        cv.putText(npimg, f"{name}", (x2 + 20, y2),
                   cv.FONT_HERSHEY_COMPLEX, 1.0, (0, 0, 255))

    return npimg


def draw_bbox(npimg, bboxes, labels):
    npimg = np.ascontiguousarray(npimg)
    if labels is None:
        labels = ["FG"] * len(bboxes)
    for bbox, label in zip(bboxes, labels):
        x1, y1, x2, y2 = map(int, bbox)
        cv.rectangle(npimg, (x1, y1), (x2, y2), (0, 0, 255), thickness=2)
        cv.putText(npimg, f"{label}", (x1, y1),
                   cv.FONT_HERSHEY_COMPLEX, 1.0, (0, 0, 255))
    return npimg


def display_image(image, output, target, save_to, names=None):
    # image, output as `Tensor[N, C, H, W]`
    image, output, target = image[0], output[0], target[0]

    if target is None:
        target = dict(id=0, bboxes=[], labels=[])

    if names is not None:
        N_DEFAULT_COLORS = len(DEFAULT_COLORS)
        colors = [DEFAULT_COLORS[i % N_DEFAULT_COLORS]
                  for i in range(len(names))]
        output = torch.argmax(output, dim=0).cpu().numpy()
        output = label2rgb(output, colors=colors, bg_label=0)
        output = draw_legend(output, names, colors)
    else:
        output = F.softmax(output, dim=0)[0].cpu().numpy()
        output = np.stack([output for _ in range(3)], 2)
        output = draw_legend(output, None, None)

    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    dtype, device = image.dtype, image.device
    mean = torch.as_tensor(mean, dtype=dtype, device=device)
    std = torch.as_tensor(std, dtype=dtype, device=device)

    if mean.ndim == 1:
        mean = mean.view(-1, 1, 1)
    if std.ndim == 1:
        std = std.view(-1, 1, 1)

    image = image.mul(std).add(mean)
    image = image.mul(255).byte()

    image = np.transpose(image.cpu().numpy(), (1, 2, 0))

    h, w, _ = image.shape
    output = cv.resize(output, (w, h), interpolation=cv.INTER_NEAREST)

    image_id = target["id"]
    bboxes = target["bboxes"]
    labels = target["labels"]
    image = draw_bbox(image, bboxes, labels)
    output = draw_bbox(output, bboxes, labels)
    img = np.concatenate((image, output), axis=1)
    filename = os.path.join(save_to, f"{image_id:04d}.jpg")
    cv.imwrite(filename, cv.cvtColor(img, cv.COLOR_RGB2BGR))
    return filename
