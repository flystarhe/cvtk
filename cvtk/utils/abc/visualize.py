import shutil
from collections import defaultdict
from pathlib import Path

import cv2 as cv
import pandas as pd
from cvtk.io import increment_path, load_json, load_pkl
from cvtk.utils.abc.nms import clean_by_bbox


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
        x, y, w, h = list(map(int, ann["bbox"]))
        cv.rectangle(img, (x, y), (x + w, y + h), color_val, thickness=2)

        if y >= 200:
            left_bottom = (x, y - offset)
        elif h >= img_h * 0.5:
            left_bottom = (x, y + h - offset)
        else:
            left_bottom = (x, y + h + 30 + offset)

        text = "{} {} {:.2f} {}/{}={:.2f}".format(
            i, ann["label"], ann.get("score", 1.0), h, w, h / w)
        cv.putText(img, text, left_bottom,
                   cv.FONT_HERSHEY_COMPLEX, 1.0, color_val)
    return img


def display_coco(coco_dir, coco_file, output_dir, **kw):
    include = kw.get("include", None)

    coco_dir = Path(coco_dir)
    coco = load_json(coco_dir / coco_file)

    output_dir = Path(output_dir)
    shutil.rmtree(output_dir, ignore_errors=True)

    include_stems = None
    if include is not None:
        include_stems = pd.read_csv(include)["file_name"].tolist()
        include_stems = set([Path(file_name).stem
                             for file_name in include_stems])

    id2name = {c["id"]: c["name"] for c in coco["categories"]}

    dataset = defaultdict(list)
    for ann in coco["annotations"]:
        ann["label"] = id2name[ann["category_id"]]
        dataset[ann["image_id"]].append(ann)

    output_dir.mkdir(parents=True)
    for img in coco["images"]:
        gts = dataset[img["id"]]
        file_name = img["file_name"]

        file_name = coco_dir / file_name

        if include_stems is not None and file_name.stem not in include_stems:
            continue

        img = cv.imread(str(file_name), 1)

        text = "target {}".format(file_name.parts[:-1][-3:])
        cv.putText(img, text, (20, 40),
                   cv.FONT_HERSHEY_COMPLEX, 1.0, (0, 0, 255))

        img = draw_bbox(img, gts, offset=10, color_val=(0, 255, 0))
        cv.imwrite(str(output_dir / file_name.name), img)
    return str(output_dir)


def display_test(results, score_thr, output_dir, **kw):
    include = kw.get("include", None)
    clean_mode = kw.get("clean_mode", None)
    clean_param = kw.get("clean_param", None)
    output_dir = increment_path(output_dir, exist_ok=False)

    if isinstance(results, str):
        shutil.copy(results, output_dir)
        results = load_pkl(results)

    include_stems = None
    if include is not None:
        include_stems = pd.read_csv(include)["file_name"].tolist()
        include_stems = set([Path(file_name).stem
                             for file_name in include_stems])

    for file_name, target, predict, dts, gts in results:
        file_name = Path(file_name)

        if include_stems is not None and file_name.stem not in include_stems:
            continue

        dts = [d for d in dts
               if d["score"] >= get_val(score_thr, d["label"], 0.3)]
        if clean_mode is not None:
            dts = clean_by_bbox(dts, clean_mode, clean_param)

        img = cv.imread(str(file_name), 1)

        text = "{} {}".format(target, file_name.parts[:-1][-3:])
        cv.putText(img, text, (20, 40),
                   cv.FONT_HERSHEY_COMPLEX, 1.0, (0, 0, 255))

        predict["bbox"] = list(map(int, predict["bbox"]))
        text = "{label} {score:.2f} {bbox}".format(**predict)
        cv.putText(img, text, (20, 80),
                   cv.FONT_HERSHEY_COMPLEX, 1.0, (0, 0, 255))

        for i, d in enumerate(dts, 1):
            d["bbox"] = list(map(int, d["bbox"]))
            text = "{} {label} {score:.2f} {bbox}".format(i, **d)
            cv.putText(img, text, (20, 80 + 40 * i),
                       cv.FONT_HERSHEY_COMPLEX, 1.0, (0, 0, 255))

        img = draw_bbox(img, dts, offset=10, color_val=(0, 0, 255))
        img = draw_bbox(img, gts, offset=50, color_val=(0, 255, 0))
        cv.imwrite(str(output_dir / file_name.name), img)
    return str(output_dir)
