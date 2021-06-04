"""Transforms for mmdet2.x"""
from collections import defaultdict

import cv2 as cv
import numpy as np

a_min, a_max = 64**2, 192**2


def _check_bboxes(src_bboxes, dst_bboxes, nonignore):
    src_w = src_bboxes[:, 2] - src_bboxes[:, 0]
    src_h = src_bboxes[:, 3] - src_bboxes[:, 1]
    src_area = src_w * src_h

    dst_w = dst_bboxes[:, 2] - dst_bboxes[:, 0]
    dst_h = dst_bboxes[:, 3] - dst_bboxes[:, 1]
    dst_area = dst_w * dst_h

    x = np.clip(src_area, a_min, a_max)
    x = (x - a_min) / (a_max - a_min)
    x = 1.0 - 0.5 * x - 1e-5

    s1 = (dst_area >= nonignore)
    s2 = (dst_area >= src_area * x)
    s3 = (dst_w >= src_w - 1) * (dst_h >= src_w * 3)
    s4 = (dst_h >= src_h - 1) * (dst_w >= src_h * 3)
    ss = s1 + s2 + s3 + s4

    inner = (dst_w >= 4) * (dst_h >= 4)
    return ss * inner, np.logical_not(ss) * inner


class RandomCrop(object):
    """Crop a random part of the input.

    Args:
        height (int): height of the crop.
        width (int): width of the crop.
        p (float): probability.

    Targets:
        image, bboxes

    Image types:
        uint8,
    """

    def __init__(self, height, width, seed=1234, **kw):
        self.base_pad = kw.get("base_pad", 16)  # >=2
        self.rng = np.random.default_rng(seed)
        area = height * width * 0.5
        self.nonignore = area
        self.height = height
        self.width = width

    def _index_selection(self, labels):
        counter = defaultdict(list)
        for index, label in enumerate(labels):
            counter[label].append(index)

        key = self.rng.choice(list(counter.keys()))
        return self.rng.choice(counter[key])

    def _get_patch(self, x_min, y_min, x_max, y_max, x_pad, y_pad):
        x0 = self.rng.integers(x_min - x_pad, x_max + x_pad - self.width)
        y0 = self.rng.integers(y_min - y_pad, y_max + y_pad - self.height)
        return x0, y0, x0 + self.width, y0 + self.height

    def _crop_and_paste(self, img, patch):
        img_h, img_w, img_c = img.shape
        x1, y1, x2, y2 = patch

        p1, x1 = (0, x1) if x1 >= 0 else (-x1, 0)
        q1, y1 = (0, y1) if y1 >= 0 else (-y1, 0)

        p2, x2 = (self.width, x2) if x2 <= img_w else (img_w - x2, img_w)
        q2, y2 = (self.height, y2) if y2 <= img_h else (img_h - y2, img_h)

        dst_img = np.zeros((self.height, self.width, img_c), dtype=img.dtype)
        dst_img[q1: q2, p1: p2] = img[y1: y2, x1: x2]
        return dst_img

    def _clip_bboxes(self, bboxes, patch):
        """
        Args:
            bboxes (ndarray): shape `(k, 4)`, dtype `np.float32`
            patch (ndarray): shape `(4,)` or tuple `(x1, y1, x2, y2)`
        """
        bboxes[:, 0::2] = np.clip(bboxes[:, 0::2], patch[0], patch[2])
        bboxes[:, 1::2] = np.clip(bboxes[:, 1::2], patch[1], patch[3])
        bboxes -= np.tile(patch[:2], 2)
        return bboxes

    def __call__(self, results):
        if self.height == 0 or self.width == 0:
            return results

        img, bboxes, labels = [results[k]
                               for k in ("img", "gt_bboxes", "gt_labels")]
        img_h, img_w, img_c = img.shape
        assert img_c == 3

        if bboxes.shape[0] == 0:
            x_min, y_min, x_max, y_max = 0, 0, img_w, img_h

            x_pad = max(self.width - img_w, 0) // 2 + self.base_pad
            y_pad = max(self.height - img_h, 0) // 2 + self.base_pad
            patch = self._get_patch(x_min, y_min, x_max, y_max, x_pad, y_pad)

            dst_img = self._crop_and_paste(img, patch)

            cx, cy = self.width // 2, self.height // 2
            x1, y1, x2, y2 = cx - 64, cy - 64, cx + 64, cy + 64

            dst_img[y1: y2, x1: x2] = (0, 0, 255)
            # set the man-made object category in 1st group
            dst_bboxes = np.array([[x1, y1, x2, y2]], dtype=np.float32)
            dst_labels = np.array([0], dtype=np.int64)

            results["img"] = dst_img
            results["img_shape"] = dst_img.shape
            results["ori_shape"] = dst_img.shape
            results["pad_shape"] = dst_img.shape
            results["gt_bboxes"] = dst_bboxes
            results["gt_labels"] = dst_labels
            return results

        index = self._index_selection(labels)
        x_min, y_min, x_max, y_max = map(int, bboxes[index])
        box_w, box_h = (x_max - x_min), (y_max - y_min)

        x_pad = max(self.width - box_w, 0) // 2 + self.base_pad
        y_pad = max(self.height - box_h, 0) // 2 + self.base_pad
        patch = self._get_patch(x_min, y_min, x_max, y_max, x_pad, y_pad)

        dst_img = self._crop_and_paste(img, patch)

        dst_bboxes = self._clip_bboxes(bboxes.copy(), patch)
        dst_mask, drop_mask = _check_bboxes(bboxes, dst_bboxes, self.nonignore)
        for x1, y1, x2, y2 in dst_bboxes[drop_mask].astype(np.int64).tolist():
            dst_img[y1: y2, x1: x2] = 0
        dst_bboxes = dst_bboxes[dst_mask]
        dst_labels = labels[dst_mask]

        results["img"] = dst_img
        results["img_shape"] = dst_img.shape
        results["ori_shape"] = dst_img.shape
        results["pad_shape"] = dst_img.shape
        results["gt_bboxes"] = dst_bboxes
        results["gt_labels"] = dst_labels
        return results


class Resize(object):

    def __init__(self, test_mode=False, img_scale=None, ratio_range=None, seed=1234, **kw):
        """Resize images & bbox.

        Args:
            img_scale (List[Tuple[int, int]]): `[(h, w)]`.
            ratio_range (Tuple[float]): `(min_ratio, max_ratio)`.
            multi_scale (List[Tuple[int, int]]): `[(src_scale, dst_scale)]`, scale with max of size.
        """
        self.test_mode = test_mode
        self.img_scale = img_scale
        self.ratio_range = ratio_range
        self.rng = np.random.default_rng(seed)
        self.multi_scale = dict(kw.get("multi_scale", []))

    def __call__(self, results):
        scale_factor = 1.0
        img = results["img"]

        src_scale = max(img.shape[:2])
        dst_scale = self.multi_scale.get(src_scale, src_scale)

        if isinstance(dst_scale, (list, tuple)):
            index = self.rng.integers(len(dst_scale))
            dst_scale = dst_scale[index]

        if src_scale != dst_scale:
            h, w = img.shape[:2]
            base_factor = scale_factor
            scale_factor = dst_scale / src_scale
            size = int(w * scale_factor + 0.5), int(h * scale_factor + 0.5)
            img = cv.resize(img, size, dst=None, interpolation=cv.INTER_LINEAR)
            scale_factor = base_factor * scale_factor

        if self.test_mode:
            results["img"] = img
            results["img_shape"] = img.shape
            results["ori_shape"] = img.shape
            results["pad_shape"] = img.shape
            results["scale_factor"] = scale_factor
            results["keep_ratio"] = True
            return results

        img_scale = self.img_scale
        if img_scale is not None:
            index = self.rng.integers(len(img_scale))
            new_h, new_w = img_scale[index]

            h, w = img.shape[:2]
            base_factor = scale_factor
            scale_factor = min(new_h / h, new_w / w)
            size = int(w * scale_factor + 0.5), int(h * scale_factor + 0.5)
            img = cv.resize(img, size, dst=None, interpolation=cv.INTER_LINEAR)
            scale_factor = base_factor * scale_factor

        ratio_range = self.ratio_range
        if ratio_range is not None:
            h, w = img.shape[:2]
            base_factor = scale_factor
            scale_factor = self.rng.uniform(*ratio_range)
            size = int(w * scale_factor + 0.5), int(h * scale_factor + 0.5)
            img = cv.resize(img, size, dst=None, interpolation=cv.INTER_LINEAR)
            scale_factor = base_factor * scale_factor

        results["img"] = img
        results["img_shape"] = img.shape
        results["ori_shape"] = img.shape
        results["pad_shape"] = img.shape
        results["scale_factor"] = scale_factor
        results["keep_ratio"] = True

        # resize bboxes
        labels = results["gt_labels"]
        bboxes = results["gt_bboxes"]
        img_shape = results["img_shape"]
        bboxes = bboxes * results["scale_factor"]
        bboxes[:, 0::2] = np.clip(bboxes[:, 0::2], 0, img_shape[1])
        bboxes[:, 1::2] = np.clip(bboxes[:, 1::2], 0, img_shape[0])
        mask = (bboxes[:, 0] < bboxes[:, 2]) * (bboxes[:, 1] < bboxes[:, 3])
        results["gt_bboxes"] = bboxes[mask]
        results["gt_labels"] = labels[mask]
        return results
