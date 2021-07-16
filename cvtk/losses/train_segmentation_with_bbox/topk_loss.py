import torch
from torch.nn import functional as F

from ._utils import _balance, _mask_top_by_full, _rescale, _sorted_and_cat


def make_target(s, feats, bboxes, labels=None, topk=3, balance=False, use_sigmoid=False, eps=1e-2):
    """Assume `cross_entropy(ignore_index=-100)`.

    Args:
        feats (Tensor[K, H, W]): the first is background.
        bboxes (List[List[int]]): such as `[[x1, y1, x2, y2],]`.
        labels (List[int], optional): where each value in `[1, K-1]`.
    """
    if labels is None:
        labels = [1] * len(bboxes)

    if use_sigmoid:
        feats = torch.sigmoid(feats)
    else:
        feats = F.softmax(feats, dim=0)

    target = torch.zeros_like(feats[0], dtype=torch.int64)
    for x1, y1, x2, y2, label in _sorted_and_cat(bboxes, labels):
        x1, y1, x2, y2 = _rescale(x1, y1, x2, y2, s)
        roi = feats[label, y1:y2, x1:x2]
        assert roi.ndim == 2

        if roi.numel() == 0:
            continue

        sub_target = torch.full_like(roi, -100, dtype=torch.int64)
        k = topk or max(roi.shape)  # is the largest edge
        selected = _mask_top_by_full(roi, k, eps)
        sub_target[selected] = label

        target[y1:y2, x1:x2] = sub_target

    if balance:
        target = _balance(target, feats[0])

    return target


def transform(pred, target, topk=3, balance=True, use_sigmoid=False, eps=1e-2):
    # `pred (Tensor[N, C, H, W])` and `target (List[Dict])`.
    s = pred.shape[-1] / target[0]["img_shape"][-1]
    pred = pred.detach()

    _target = []
    for i in range(pred.shape[0]):
        _target.append(make_target(s, pred[i], target[i]["bboxes"], target[i]["labels"],
                                   topk, balance, use_sigmoid, eps))
    return torch.stack(_target, 0)


def criterion(inputs, target, topk=3, balance=True, use_sigmoid=False, eps=1e-2):
    """
    Args:
        inputs (OrderedDict): required `out`, optional `aux`.
        target (List[Dict]): required `bboxes`, `img_shape`, `labels`.
    """
    _pred = inputs["out"]
    _target = transform(_pred, target, topk, balance, use_sigmoid, eps)
    if use_sigmoid:
        raise NotImplementedError("Not Implemented ...")
    else:
        loss = F.cross_entropy(_pred, _target, ignore_index=-100)

    if "aux" in inputs:
        _pred = inputs["aux"]
        _target = transform(_pred, target, topk, balance, use_sigmoid, eps)
        if use_sigmoid:
            raise NotImplementedError("Not Implemented ...")
        else:
            return loss + 0.5 * F.cross_entropy(_pred, _target, ignore_index=-100)

    return loss
