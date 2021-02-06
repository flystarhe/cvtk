import hiplot as hip
import numpy as np

from cvtk.io import load_json
from cvtk.io import load_pkl
from cvtk.utils.abc import nms


def get_val(data, key, val=None):
    if key in data:
        return data[key]
    if "*" in data:
        return data["*"]
    return val


def best_iou(w, h, anchors, ratios):
    data = []
    for anchor in anchors:
        for ratio in ratios:
            c = np.sqrt(ratio)
            anchor_w = anchor / c
            anchor_h = anchor * c
            I = min(w, anchor_w) * min(h, anchor_h)
            U = (w * h) + (anchor ** 2) - I
            data.append(I / U)
    return max(data)


def hip_bbox(coco_file, crop_size, scales=[8], ratios=[0.5, 1.0, 2.0], base_sizes=[4, 8, 16, 32, 64]):
    anchors = [s * x for s in scales for x in base_sizes]

    coco = load_json(coco_file)
    imgs = {img["id"]: img["file_name"] for img in coco["images"]}

    data = []
    for ann in coco["annotations"]:
        w, h = [min(x, crop_size) for x in ann["bbox"][2:]]
        data.append({"file_name": imgs[ann["image_id"]], "id": ann["category_id"], "iou": best_iou(w, h, anchors, ratios),
                     "h_ratio": h / w, "h_ratio_log2": np.log2(h / w),
                     "area": w * h, "min_size": min(w, h)})

    hip.Experiment.from_iterable(data).display()
    return "jupyter.hiplot"


def hip_result(results, score_thr, **kw):
    """Show model prediction results, allow gt is empty.

    Args:
        results (list): List of `(file_name, none, none, dt, gt)`.
        score_thr (dict): Such as `dict(CODE1=S1, CODE2=S2, ...)`.
    """
    clean_mode = kw.get("clean_mode", "min")
    clean_param = kw.get("clean_param", 0.1)
    match_mode = kw.get("match_mode", "iou")
    min_pos_iou = kw.get("min_pos_iou", 0.3)

    if isinstance(results, str):
        results = load_pkl(results)

    vals = []
    for file_name, _, _, dt, gt in results:
        dt = [d for d in dt if d["score"] >= get_val(score_thr, d["label"], 0.3)]
        dt = nms.clean_by_bbox(dt, clean_mode, clean_param)
        ious = nms.bbox_overlaps(dt, gt, match_mode)

        exclude_i = set()
        exclude_j = set()
        if ious is not None:
            for i, j in enumerate(ious.argmax(axis=1)):
                iou = ious[i, j]
                d, g = dt[i], gt[j]
                if iou >= min_pos_iou:
                    a = [d["label"], d["score"]] + d["bbox"][2:]
                    b = [g["label"], g["score"]] + g["bbox"][2:]
                    vals.append([file_name, iou] + a + b)
                    exclude_i.add(i)
                    exclude_j.add(j)

        for i, d in enumerate(dt):
            d = dt[i]
            if i not in exclude_i:
                a = [d["label"], d["score"]] + d["bbox"][2:]
                b = ["none", 0., 0, 0]
                vals.append([file_name, 0.] + a + b)

        for j, g in enumerate(gt):
            g = gt[j]
            if j not in exclude_j:
                a = ["none", 0., 0, 0]
                b = [g["label"], g["score"]] + g["bbox"][2:]
                vals.append([file_name, 0.] + a + b)

    names = "file_name,iou,label,score,w,h,gt_label,gt_score,gt_w,gt_h".split(",")
    data = [{a: b for a, b in zip(names, val)} for val in vals]
    hip.Experiment.from_iterable(data).display()
    return "jupyter.hiplot"
