import shutil
from collections import defaultdict
from pathlib import Path

import cv2 as cv
import pandas as pd
from cvtk.io import increment_path, load_json, load_pkl
from cvtk.utils.abc.nms import clean_by_bbox


def load_csv_files(csv_files):
    set_out, set_in = set(), set()

    if csv_files is None:
        return set_out, set_in

    for f in csv_files.split(","):
        if ".csv" not in f:
            continue

        if f[0] == "-":
            dat = pd.read_csv(f[1:])["file_name"].tolist()
            set_out.update([Path(file_name).stem for file_name in dat])
        else:
            dat = pd.read_csv(f)["file_name"].tolist()
            set_in.update([Path(file_name).stem for file_name in dat])

    return set_out, set_in


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
    level = kw.get("level", 1)
    filters = kw.get("filters", None)
    group_by = kw.get("group_by", None)

    coco_dir = Path(coco_dir)
    coco = load_json(coco_dir / coco_file)

    output_dir = Path(output_dir)
    shutil.rmtree(output_dir, ignore_errors=True)

    set_out, set_in = load_csv_files(filters)

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
        target = Path(file_name).parent.name

        if set_out and file_name.stem in set_out:
            continue
        if set_in and file_name.stem not in set_in:
            continue

        img = cv.imread(str(file_name), 1)

        text = "{} {}".format(target, file_name.parts[:-1][-level:])
        cv.putText(img, text, (20, 40),
                   cv.FONT_HERSHEY_COMPLEX, 1.0, (0, 0, 255))

        img = draw_bbox(img, gts, offset=10, color_val=(0, 255, 0))

        dst_dir = output_dir / "images"
        if group_by == "target":
            dst_dir = output_dir / str(target)

        dst_dir.mkdir(parents=True, exist_ok=True)
        cv.imwrite(str(dst_dir / file_name.name), img)
    return str(output_dir)


def display_test(results, score_thr, output_dir, **kw):
    level = kw.get("level", 1)
    filters = kw.get("filters", None)
    group_by = kw.get("group_by", None)
    clean_mode = kw.get("clean_mode", None)
    clean_param = kw.get("clean_param", None)
    output_dir = increment_path(output_dir, exist_ok=False)

    if isinstance(results, str):
        shutil.copy(results, output_dir)
        results = load_pkl(results)

    set_out, set_in = load_csv_files(filters)

    for file_name, target, predict, dts, gts in results:
        file_name = Path(file_name)

        if set_out and file_name.stem in set_out:
            continue
        if set_in and file_name.stem not in set_in:
            continue

        dts = [d for d in dts
               if d["score"] >= get_val(score_thr, d["label"], 0.3)]
        if clean_mode is not None:
            dts = clean_by_bbox(dts, clean_mode, clean_param)

        img = cv.imread(str(file_name), 1)

        text = "{} {}".format(target, file_name.parts[:-1][-level:])
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

        dst_dir = output_dir / "images"
        if group_by == "target":
            dst_dir = output_dir / str(target)
        elif group_by == "predict":
            dst_dir = output_dir / predict["label"]

        dst_dir.mkdir(parents=True, exist_ok=True)
        cv.imwrite(str(dst_dir / file_name.name), img)
    return str(output_dir)
