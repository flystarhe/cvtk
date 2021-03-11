import math

import torch
from torch import nn
from torch.nn import functional as F


def _split(a, b, n):
    n = min(n, b - a)
    x = torch.linspace(a, b, n + 1, dtype=torch.int64).tolist()
    return [(i, j) for i, j in zip(x, x[1:])]


def _point(feat, topk, x1, y1, x2, y2):
    _y_pairs = _split(y1, y2, topk)
    _x_pairs = _split(x1, x2, topk)

    points = []
    for _y1, _y2 in _y_pairs:
        for _x1, _x2 in _x_pairs:
            _shift = torch.argmax(feat[_y1:_y2, _x1:_x2]).item()
            y_shift, x_shift = divmod(_shift, _x2 - _x1)
            cy, cx = _y1 + y_shift, _x1 + x_shift
            cs = feat[cy, cx].item()  # sort key
            points.append((cy, cx, cs))

    topk = max(len(_y_pairs), len(_x_pairs))
    return sorted(points, key=lambda args: args[2])[-topk:]


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
        for cy, cx, _ in _point(feats[label], topk, x1, y1, x2, y2):
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


def test():
    s = 1.0
    topk = 3
    feats = torch.randn(3, 14, 14)
    bboxes = [[1, 1, 9, 5], [3, 3, 13, 7], [9, 9, 13, 13]]
    labels = [1, 2, 1]
    res = make_target(s, topk, feats, bboxes, labels)
    res[res.eq(-100)] = -1
    return res
