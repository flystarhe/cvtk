import hiplot as hip
import numpy as np
import pandas as pd
from cvtk.io import load_json, load_pkl
from cvtk.utils.abc.nms import bbox_overlaps


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
        return {f"L{i}": p for i, p in enumerate(parts, 1)}

    return [{**d, **_split(d["file_name"])} for d in data]


def hip_coco(coco_file, crop_size, splits=0, scales=[8], base_sizes=[4, 8, 16, 32, 64], ratios=[0.5, 1.0, 2.0], silent=False):
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
                     "h_ratio": h / w, "scale": np.sqrt(h * w),
                     "max_size": max(w, h), "min_size": min(w, h)})

    if splits > 0:
        data = split_file_name(data, splits)

    if silent:
        return data

    hip.Experiment.from_iterable(data).display()
    return "jupyter.hiplot"


def hip_test(results, splits=0, score_thr=None, match_mode="iou", min_pos_iou=0.25, silent=False):
    """Show model prediction results, allow gts is empty.

    Args:
        results (list): List of `tuple(img_path, target, predict, dts, gts)`
        score_thr (dict): Such as `{"CODE1":S1, "CODE2":S2, "*":0.3}`
    """
    if isinstance(results, str):
        results = load_pkl(results)

    if score_thr is None:
        score_thr = {"*": 0.3}

    vals = []
    for file_name, target, predict, dts, gts in results:
        dts = [dt for dt in dts
               if dt["score"] >= get_val(score_thr, dt["label"], 0.3)]
        ious = bbox_overlaps(dts, gts, match_mode)
        n_gt, n_dt = len(gts), len(dts)

        base_info = [file_name, target, predict["label"], predict["score"]]

        exclude_i = set()
        exclude_j = set()
        if ious is not None:
            for i, j in enumerate(ious.argmax(axis=1)):
                iou = float(ious[i, j])
                dt, gt = dts[i], gts[j]
                is_ok = "Y" if dt["label"] == gt["label"] else "N"
                if iou >= min_pos_iou:
                    a = [dt["label"], dt["score"], dt["area"]] + dt["bbox"][2:]
                    b = [gt["label"], gt["score"], gt["area"]] + gt["bbox"][2:]
                    vals.append(base_info + [iou, is_ok, n_gt, n_dt] + a + b)
                    exclude_i.add(i)
                    exclude_j.add(j)

        iou = 0.
        is_ok = "N"

        for i, dt in enumerate(dts):
            dt = dts[i]
            if i not in exclude_i:
                a = [dt["label"], dt["score"], dt["area"]] + dt["bbox"][2:]
                b = ["none", 0., 1, 1, 1]
                vals.append(base_info + [iou, is_ok, n_gt, n_dt] + a + b)

        for j, gt in enumerate(gts):
            gt = gts[j]
            if j not in exclude_j:
                a = ["none", 0., 1, 1, 1]
                b = [gt["label"], gt["score"], gt["area"]] + gt["bbox"][2:]
                vals.append(base_info + [iou, is_ok, n_gt, n_dt] + a + b)

    names = ["file_name", "t_label", "p_label", "p_score",
             "iou", "is_ok", "n_gt", "n_dt",
             "label", "score", "area", "w", "h",
             "gt_label", "gt_score", "gt_area", "gt_w", "gt_h"]
    data = [{a: b for a, b in zip(names, val)} for val in vals]

    if splits > 0:
        data = split_file_name(data, splits)

    if silent:
        return data

    hip.Experiment.from_iterable(data).display()
    return "jupyter.hiplot"


def hip_test_image(results, splits=0, silent=False):
    """Show model prediction results, allow gts is empty.

    Args:
        results (list): List of `tuple(img_path, target, predict, dts, gts)`
    """
    if isinstance(results, str):
        results = load_pkl(results)

    vals = []
    for file_name, target, predict, dts, gts in results:
        is_ok = "Y" if target == predict["label"] else "N"
        vals.append([file_name, target, len(gts),
                     len(dts), predict["label"], predict["score"], is_ok])

    names = ["file_name", "t_label", "n_gt",
             "n_dt", "p_label", "p_score", "is_ok"]
    data = [{a: b for a, b in zip(names, val)} for val in vals]

    if splits > 0:
        data = split_file_name(data, splits)

    if silent:
        return data

    hip.Experiment.from_iterable(data).display()
    return "jupyter.hiplot"


def hardmini_test(logs, level="image", score=0.85, nok=True):
    pkl_list = [line.strip() for line in logs if ".pkl" in line]
    assert len(pkl_list) == 1, f"must be one and only one: {pkl_list}"

    pkl_file = pkl_list[0]
    if level == "image":
        data = hip_test_image(pkl_file, splits=0, silent=True)
        if nok:
            data = [d for d in data
                    if d["p_score"] < score or d["is_ok"] == "N"]
        else:
            data = [d for d in data if d["p_score"] < score]
    elif level == "object":
        data = hip_test(pkl_file, splits=0, silent=True)
        if nok:
            data = [d for d in data
                    if d["score"] < score or d["is_ok"] == "N"]
        else:
            data = [d for d in data if d["score"] < score]
    else:
        data = [{"file_name": "none"}]

    flag = f"_{level}_{score:.2f}.csv"

    f = pkl_file + flag
    df = pd.DataFrame(data)
    df.to_csv(f, index=False)
    return f"{f}, {df.shape[0]}"
