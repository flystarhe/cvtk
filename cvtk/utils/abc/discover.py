import hiplot as hip
import numpy as np
from cvtk.io import load_json, load_pkl
from cvtk.utils.abc.gen import image_label
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


def split_file_name(data, n=1):
    if "file_name" not in data[0]:
        return data

    def _split(file_name):
        parts = file_name.split("/")[:-1][-n:]
        return {f"p{i}": p for i, p in enumerate(parts, 1)}

    return [{**d, **_split(d["file_name"])} for d in data]


def hip_coco(coco_file, crop_size, scales=[8], ratios=[0.5, 1.0, 2.0], base_sizes=[4, 8, 16, 32, 64], splits=0):
    anchors = [s * x for s in scales for x in base_sizes]

    coco = load_json(coco_file)
    cats = {cat["id"]: cat["name"] for cat in coco["categories"]}
    imgs = {img["id"]: img["file_name"] for img in coco["images"]}

    data = []
    for ann in coco["annotations"]:
        w, h = [min(x, crop_size) for x in ann["bbox"][2:]]
        data.append({"label": cats[ann["category_id"]],
                     "file_name": imgs[ann["image_id"]],
                     "iou": best_iou(w, h, anchors, ratios),
                     "h_ratio": h / w, "h_ratio_log2": np.log2(h / w),
                     "area": w * h, "min_wh": min(w, h)})

    if splits > 0:
        data = split_file_name(data, splits)
    hip.Experiment.from_iterable(data).display()
    return "jupyter.hiplot"


def hip_test(results, score_thr, splits=0, **kw):
    """Show model prediction results, allow gts is empty.

    Args:
        results (list): List of `(img_path, target, predict, dts, gts)`.
        score_thr (dict): Such as `dict(CODE1=S1, CODE2=S2, ...)`.
    """
    params = dict(mode="complex", score_thr={"*": 0.3}, label_grade={"*": 1})
    clean_mode = kw.get("clean_mode", "min")
    clean_param = kw.get("clean_param", 0.1)
    match_mode = kw.get("match_mode", "iou")
    min_pos_iou = kw.get("min_pos_iou", 0.3)
    by_image = kw.get("by_image", False)
    kw_gen = kw.get("kw_gen", params)

    if isinstance(results, str):
        results = load_pkl(results)

    vals = []
    for file_name, target, predict, dts, gts in results:
        if by_image:
            predict = image_label(dts, **kw_gen)
            vals.append(
                [file_name, target, predict["label"], predict["score"]])
            continue

        dts = [dt for dt in dts
               if dt["score"] >= get_val(score_thr, dt["label"], 0.3)]
        dts = nms.clean_by_bbox(dts, clean_mode, clean_param)
        ious = nms.bbox_overlaps(dts, gts, match_mode)

        exclude_i = set()
        exclude_j = set()
        if ious is not None:
            for i, j in enumerate(ious.argmax(axis=1)):
                iou = ious[i, j]
                dt, gt = dts[i], gts[j]
                if iou >= min_pos_iou:
                    a = [dt["label"], dt["score"]] + dt["bbox"][2:]
                    b = [gt["label"], gt["score"]] + gt["bbox"][2:]
                    vals.append([file_name, iou] + a + b)
                    exclude_i.add(i)
                    exclude_j.add(j)

        for i, dt in enumerate(dts):
            dt = dts[i]
            if i not in exclude_i:
                a = [dt["label"], dt["score"]] + dt["bbox"][2:]
                b = ["none", 0., 1, 1]
                vals.append([file_name, 0.] + a + b)

        for j, gt in enumerate(gts):
            gt = gts[j]
            if j not in exclude_j:
                a = ["none", 0., 1, 1]
                b = [gt["label"], gt["score"]] + gt["bbox"][2:]
                vals.append([file_name, 0.] + a + b)

    if by_image:
        names = ["file_name", "target", "label", "score"]
    else:
        names = ["file_name", "iou",
                 "label", "score", "w", "h",
                 "gt_label", "gt_score", "gt_w", "gt_h"]
    data = [{a: b for a, b in zip(names, val)} for val in vals]

    if splits > 0:
        data = split_file_name(data, splits)
    hip.Experiment.from_iterable(data).display()
    return "jupyter.hiplot"
