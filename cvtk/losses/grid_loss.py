import math
import torch
from torch.nn import functional as F


def _split(a, b, n=7):
    n = max(1, min(n, (b - a) // 2))
    x = torch.linspace(a, b, n + 1, dtype=torch.int64).tolist()
    return [(i, j) for i, j in zip(x, x[1:])]


def _point(feat, topk, x1, y1, x2, y2):
    _y_pairs = _split(y1, y2, topk)
    _x_pairs = _split(x1, x2, topk)
    topk = max(len(_y_pairs), len(_x_pairs))

    points = []
    for _y1, _y2 in _y_pairs:
        for _x1, _x2 in _x_pairs:
            _shift = torch.argmax(feat[_y1:_y2, _x1:_x2]).item()
            _shift_y, _shift_x = divmod(_shift, _x2 - _x1)
            cy, cx = _y1 + _shift_y, _x1 + _shift_x
            cs = feat[cy, cx].item()  # sort key
            points.append((cy, cx, cs))

    return sorted(points, key=lambda args: args[2], reverse=True)[:topk]


def balance_target(target, weight):
    """Assume `cross_entropy(ignore_index=-100)`.

    Args:
        target (Tensor[H, W]): the input tensor.
        weight (Tensor[H, W]): probability of background.
    """
    negative_mask = target.eq(0)
    n_positive = target.gt(0).sum().item()
    limit = max(target.size(1), n_positive * 2)

    n_negative = negative_mask.sum().item()
    if n_negative >= limit + 2:
        n = (n_negative - limit) // 2
        probs = weight[negative_mask].sort()[0]
        target[negative_mask * weight.lt(probs[n])] = -100
        target[negative_mask * weight.gt(probs[-1-n])] = -100

    return target


def make_target(s, topk, feats, bboxes, labels=None, balance=False):
    """Assume `cross_entropy(ignore_index=-100)`.

    Args:
        feats (Tensor[K, H, W]): the first is background.
        bboxes (List[List[int]]): such as `[[x1, y1, x2, y2],]`.
        labels (List[int], optional): where each value in `[1, K-1]`.
    """
    _, h, w = feats.size()
    feats = F.softmax(feats, dim=0)
    target = torch.zeros(h, w, dtype=torch.int64, device=feats.device)

    if labels is None:
        labels = [1] * len(bboxes)

    data = [(x1, y1, x2, y2, label, (x2 - x1) * (y2 - y1)) for (x1, y1, x2, y2), label in zip(bboxes, labels)]
    for x1, y1, x2, y2, label, _ in sorted(data, key=lambda args: args[5], reverse=True):
        x1 = math.floor(x1 * s)
        y1 = math.floor(y1 * s)
        x2 = math.ceil(x2 * s) + 1
        y2 = math.ceil(y2 * s) + 1
        x2, y2 = min(w, x2), min(h, y2)

        target[y1:y2, x1:x2] = -100
        for cy, cx, _ in _point(feats[label], topk, x1, y1, x2, y2):
            target[cy, cx] = label

    if balance:
        target = balance_target(target, feats[0])

    return target


def test():
    s = 1.0
    topk = 3
    feats = torch.randn(3, 14, 14)
    bboxes = [[1, 1, 9, 5], [3, 3, 13, 7], [9, 9, 13, 13]]
    labels = [1, 2, 1]
    res = make_target(s, topk, feats, bboxes, labels)
    res[res.eq(-100)] = -1
    return res
