import math

import torch
from torch import nn
from torch.nn import functional as F


def _point(feat, topk, x1, y1, x2, y2):
    if topk == 1:
        _shift = torch.argmax(feat[y1:y2, x1:x2]).item()
        y_shift, x_shift = divmod(_shift, x2 - x1)
        return [(y1 + y_shift, x1 + x_shift)]

    tensor = feat[y1:y2, x1:x2].contiguous()
    _, indices = tensor.view(-1).sort()
    indices = indices.tolist()
    w, h = x2 - x1, y2 - y1
    topk = max(w, h)

    points = []
    for _shift in indices[-topk:]:
        y_shift, x_shift = divmod(_shift, w)
        points.append((y1 + y_shift, x1 + x_shift))

    return points


def _balance(target, weight):
    """Assume `cross_entropy(ignore_index=-100)`.

    Args:
        target (Tensor[H, W]): the input tensor.
        weight (Tensor[H, W]): probability of background.
    """
    negative_mask = target.eq(0)
    n_positive = target.gt(0).sum().item()

    n_negative = negative_mask.sum().item()
    limit = max(target.size(1), n_positive * 3)
    if n_negative > limit:
        p = weight[negative_mask].sort()[0]
        target[negative_mask * weight.gt(p[limit])] = -100

    return target


def make_target(s, topk, feats, bboxes, labels=None, balance=False):
    """Assume `cross_entropy(ignore_index=-100)`.

    Args:
        feats (Tensor[K, H, W]): the first is background.
        bboxes (List[List[int]]): such as `[[x1, y1, x2, y2],]`.
        labels (List[int], optional): where each value in `[1, K-1]`.
    """
    if labels is None:
        labels = [1] * len(bboxes)

    _, h, w = feats.size()
    feats = F.softmax(feats, dim=0)
    masks = torch.zeros_like(feats, dtype=torch.uint8)

    data = [(x1, y1, x2, y2, label, (x2 - x1) * (y2 - y1)) for (x1, y1, x2, y2), label in zip(bboxes, labels)]
    for x1, y1, x2, y2, label, _ in sorted(data, key=lambda args: args[5], reverse=True):
        x1 = math.floor(x1 * s + 0.3)
        y1 = math.floor(y1 * s + 0.3)
        x2 = math.ceil(x2 * s - 0.3) + 1
        y2 = math.ceil(y2 * s - 0.3) + 1
        x2, y2 = min(w, x2), min(h, y2)

        masks[label, y1:y2, x1:x2] = 2
        for cy, cx in _point(feats[label], topk, x1, y1, x2, y2):
            masks[label, cy, cx] = 1

    target = masks.argmax(0)

    mask = masks.sum(0)
    target[mask == 0] = 0
    target[mask >= 2] = -100

    if balance:
        target = _balance(target, feats[0])

    return target


def transform(pred, target, topk=3, balance=True):
    # where `pred` type as `Tensor[N, C, H, W]`.
    s = pred.size(-1) / target[0]["img_shape"][-1]
    pred = pred.detach()

    _target = []
    for i in range(pred.size(0)):
        _target.append(make_target(s, topk, pred[i], target[i]["bboxes"], target[i]["labels"], balance))
    return torch.stack(_target, 0)


def criterion(inputs, target, topk=3, balance=True):
    """
    Args:
        inputs (OrderedDict): required `out`. optional `aux`.
        target (List[Dict]): required `bboxes`, `img_shape`, `labels`.
    """
    _pred = inputs["out"]
    _target = transform(_pred, target, topk, balance)
    loss = nn.functional.cross_entropy(_pred, _target, ignore_index=-100)

    if "aux" in inputs:
        _pred = inputs["aux"]
        _target = transform(_pred, target, topk, balance)
        return loss + 0.5 * nn.functional.cross_entropy(_pred, _target, ignore_index=-100)

    return loss
