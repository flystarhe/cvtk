import os

import cv2 as cv
import numpy as np
import torch
from torch.nn import functional as F


def draw_bbox(npimg, bboxes, labels):
    if labels is None:
        labels = ["1"] * len(bboxes)
    npimg = np.ascontiguousarray(npimg)
    for bbox, label in zip(bboxes, labels):
        x1, y1, x2, y2 = list(map(int, bbox))
        cv.rectangle(npimg, (x1, y1), (x2, y2), (0, 0, 255), thickness=2)
        cv.putText(img, f"{label}", (x1, y1), cv.FONT_HERSHEY_COMPLEX, 1.0, (0, 0, 255))
    return npimg


def display_image(image, output, target, save_to):
    # image, output as `Tensor[N, C, H, W]`
    image, output, target = image[0], output[0], target[0]
    output = F.softmax(output, dim=0)[0]  # p of prediction as background

    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    dtype = image.dtype
    mean = torch.as_tensor(mean, dtype=dtype, device=image.device)
    std = torch.as_tensor(std, dtype=dtype, device=image.device)

    if mean.ndim == 1:
        mean = mean.view(-1, 1, 1)
    if std.ndim == 1:
        std = std.view(-1, 1, 1)

    image = image.mul(std).add(mean)
    image = image.mul(255).byte()

    output = torch.stack([output for _ in range(3)], 0)
    output = output.mul(255).byte()

    image = np.transpose(image.cpu().numpy(), (1, 2, 0))
    output = np.transpose(output.cpu().numpy(), (1, 2, 0))

    h, w, _ = image.shape
    output = cv.resize(output, (w, h), interpolation=cv.INTER_NEAREST)

    image_id = target["id"]
    bboxes = target["bboxes"]
    image = draw_bbox(image, bboxes, None)
    output = draw_bbox(output, bboxes, None)
    img = np.concatenate((image, output), axis=1)
    filename = os.path.join(save_to, f"{image_id:04d}.jpg")
    cv.imwrite(filename, cv.cvtColor(img, cv.COLOR_RGB2BGR))
    return filename
