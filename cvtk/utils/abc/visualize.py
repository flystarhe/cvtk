import cv2 as cv
import pandas as pd
import shutil
from collections import defaultdict
from pathlib import Path


from cvtk.io import increment_path
from cvtk.io import load_json
from cvtk.io import load_pkl
from cvtk.utils.abc import nms


def get_val(data, key, val=None):
    if key in data:
        return data[key]
    if "*" in data:
        return data["*"]
    return val


def draw_bbox(img, anns, offset=0, color_val=(0, 0, 255)):
    if isinstance(img, (str, Path)):
        img = cv.imread(str(img), 1)

    img_h, img_w = img.shape[:2]
    for i, ann in enumerate(anns, 1):
        x, y, w, h = map(int, ann["bbox"])
        cv.rectangle(img, (x, y), (x + w, y + h), color_val, thickness=2)

        if y >= 60:
            left_bottom = (x, y - offset)
        elif h >= img_h * 0.5:
            left_bottom = (x, y + h - offset)
        else:
            left_bottom = (x, y + h + 30 + offset)

        text = "{}: {}: {:.2f}: {}/{}={:.2f}".format(
            i, ann["label"], ann.get("score", 1.0), h, w, h / w)
        cv.putText(img, text, left_bottom, cv.FONT_HERSHEY_COMPLEX, 1.0, color_val)
    return img


def display_coco(coco_dir, coco_file, output_dir, **kw):
    coco_dir = Path(coco_dir)
    coco = load_json(coco_dir / coco_file)

    output_dir = Path(output_dir)
    shutil.rmtree(output_dir, ignore_errors=True)

    output_dir.mkdir(parents=True)

    include = kw.get("include", None)

    targets = None
    if include is not None:
        targets = pd.read_csv(include)["file_name"].tolist()
        targets = set([Path(file_name).stem for file_name in targets])

    id2name = {c["id"]: c["name"] for c in coco["categories"]}

    cache = defaultdict(list)
    for ann in coco["annotations"]:
        ann["label"] = id2name[ann["category_id"]]
        cache[ann["image_id"]].append(ann)

    gts, imgs = [], []
    for img in coco["images"]:
        gts.append(cache[img["id"]])
        imgs.append(coco_dir / img["file_name"])

    for anns, file_name in zip(gts, imgs):
        file_name = Path(file_name)

        if targets is not None and file_name.stem not in targets:
            continue

        img = draw_bbox(file_name, anns, offset=0, color_val=(0, 255, 0))
        cv.imwrite(str(output_dir / file_name.name), img)
    return str(output_dir)


def display_result(results, score_thr, output_dir, **kw):
    simple = kw.get("simple", True)
    include = kw.get("include", None)
    clean_mode = kw.get("clean_mode", "min")
    clean_param = kw.get("clean_param", 0.1)
    output_dir = increment_path(output_dir, exist_ok=False)

    if isinstance(results, str):
        shutil.copy(results, output_dir)
        results = load_pkl(results)

    targets = None
    if include is not None:
        targets = pd.read_csv(include)["file_name"].tolist()
        targets = set([Path(file_name).stem for file_name in targets])

    for file_name, _, _, dt, gt in results:
        file_name = Path(file_name)

        if targets is not None and file_name.stem not in targets:
            continue

        dt = [d for d in dt if d["score"] >= get_val(score_thr, d["label"], 0.3)]
        if simple:
            dt = nms.clean_by_bbox(dt, clean_mode, clean_param)

        img = draw_bbox(file_name, dt, offset=0, color_val=(0, 0, 255))
        img = draw_bbox(img, gt, offset=30, color_val=(0, 255, 0))
        cv.imwrite(str(output_dir / file_name.name), img)
    return str(output_dir)
