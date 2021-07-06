import math

import torch
from torch import Tensor, nn
from torch.nn import functional as F


def _mask_top_by_full(roi: Tensor, k: int = 1, eps: float = 1e-2):
    """Masked best k sub_roi/pixel.

    Args:
        roi (Tensor[H, W]): input tensor.
    """
    flattened = torch.flatten(roi, start_dim=0, end_dim=-1)
    val = torch.topk(flattened, k, dim=-1)[0][-1]
    return torch.gt(roi, val - eps)


def _mask_top_by_grid(roi: Tensor, k: int = 1, eps: float = 1e-2):
    """Masked best k sub_roi/pixel.

    Args:
        roi (Tensor[H, W]): input tensor.
    """
    kernel_size = tuple([(s - 1) // k + 1 for s in roi.shape])

    if kernel_size > (1, 1):
        pool = nn.MaxPool2d(kernel_size, ceil_mode=True)

        output = pool(roi[None, None])
        flattened = torch.flatten(output)
        val = torch.topk(flattened, max(output.shape), dim=-1)[0][-1]
        # output = F.interpolate(output, size=roi.shape, align_corners=False)
        return torch.gt(roi, val - eps)

    return _mask_top_by_full(roi, k=max(roi.shape), eps=eps)


def _mask_top_by_line_h(roi: Tensor, k: int = 1, eps: float = 1e-2):
    """Masked best k sub_roi/pixel.

    Args:
        roi (Tensor[H, W]): input tensor.
    """
    mat = torch.topk(roi, k, dim=0)[0][-1:, :]
    return torch.gt(roi, mat - eps)


def _mask_top_by_line_w(roi: Tensor, k: int = 1, eps: float = 1e-2):
    """Masked best k sub_roi/pixel.

    Args:
        roi (Tensor[H, W]): input tensor.
    """
    mat = torch.topk(roi, k, dim=1)[0][:, -1:]
    return torch.gt(roi, mat - eps)


def _mask_top_by_line(roi: Tensor, k: int = 1, eps: float = 1e-2):
    """Masked best k sub_roi/pixel.

    Args:
        roi (Tensor[H, W]): input tensor.
    """
    a = _mask_top_by_line_h(roi, k=k, eps=eps)
    b = _mask_top_by_line_w(roi, k=k, eps=eps)
    return a + b


def _sorted_and_cat(bboxes, labels):
    """Sorted by box area.

    Args:
        bboxes (List[List[int]]): such as `[[x1, y1, x2, y2],]`.
        labels (List[int]): where each value in `[1, C]`, `0` is BG.
    """
    dat = [(x1, y1, x2, y2, label, (x2 - x1) * (y2 - y1))
           for (x1, y1, x2, y2), label in zip(bboxes, labels)]
    return [v[:-1] for v in sorted(dat, key=lambda v: v[-1], reverse=True)]


def _rescale(x1, y1, x2, y2, factor):
    """Rescale a bbox by factor.

    Args:
        x1, y1, x2, y2: (int, float) from bboxes.
        factor (float): is `1 / stride`.
    """
    x1 = math.floor(x1 * factor + 0.3)
    y1 = math.floor(y1 * factor + 0.3)
    x2 = math.floor(x2 * factor + 0.7)
    y2 = math.floor(y2 * factor + 0.7)
    return x1, y1, max(x1 + 1, x2), max(y1 + 1, y2)


def _balance(target, weight):
    """Assume `cross_entropy(ignore_index=-100)`.

    Args:
        target (Tensor[H, W]): input tensor.
        weight (Tensor[H, W]): probability of BG.
    """
    negative_mask = target.eq(0)
    limit = 3 + 3 * target.gt(0).sum().item()
    if negative_mask.sum().item() > limit:
        p = weight[negative_mask].sort()[0][limit]
        target[negative_mask * weight.gt(p)] = -100
    return target
