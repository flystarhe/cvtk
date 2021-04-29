import shutil
from pathlib import Path
from xml.etree import ElementTree

import cv2 as cv
import numpy as np
import pandas as pd
from cvtk.io import load_json, save_json
from cvtk.utils.abc import nms
from lxml import etree

DEL_LABELS = set(["__DEL"])
ANN_EXTENSIONS = set([".xml", ".json"])
IMG_EXTENSIONS = set([".jpg", ".jpeg", ".png", ".bmp"])


def _trans(data, key):
    if key in data:
        return data[key]
    if "*" in data:
        return data["*"]
    return key


def _norm(bbox, img_w, img_h):
    x, y, w, h = map(int, bbox)
    x, y = max(0, x), max(0, y)
    w = min(img_w - x, w)
    h = min(img_h - y, h)
    return [x, y, w, h]


def _filter(img_dir, ann_dir, include=None):
    img_list = sorted(Path(img_dir).glob("**/*"))
    img_list = [x for x in img_list if x.suffix in IMG_EXTENSIONS]

    if include is not None:
        include = Path(include)
        if include.is_dir():
            targets = [x for x in include.glob("**/*")]
        elif include.suffix == ".csv":  # from hiplot
            targets = pd.read_csv(include)["file_name"].tolist()
        elif include.suffix == ".json":  # from coco dataset
            targets = [x["file_name"] for x in load_json(include)["images"]]
        else:
            raise NotImplementedError(f"Not Implemented: {include.name}")

        targets = set([Path(file_name).stem for file_name in targets])
        img_list = [x for x in img_list if x.stem in targets]

    imgs = {x.stem: x for x in img_list}

    if ann_dir is None:
        ann_list = img_list
    else:
        ann_list = sorted(Path(ann_dir).glob("**/*"))
        ann_list = [x for x in ann_list if x.suffix in ANN_EXTENSIONS]

    anns = {x.stem: x for x in ann_list}

    ks = set(imgs.keys()) & set(anns.keys())
    data = [(imgs[k], anns[k]) for k in sorted(ks)]
    print(f"filter {len(data)} from imgs/anns({len(imgs)}/{len(anns)})")
    return data


def xml2json(xml_path):
    xml_path = Path(xml_path)
    parser = etree.XMLParser(encoding="utf-8")
    xmltree = ElementTree.parse(xml_path, parser=parser).getroot()

    shapes = []
    for object_iter in xmltree.findall("object"):
        name = object_iter.find("name").text
        bndbox = object_iter.find("bndbox")
        xmin = int(float(bndbox.find("xmin").text))
        ymin = int(float(bndbox.find("ymin").text))
        xmax = int(float(bndbox.find("xmax").text))
        ymax = int(float(bndbox.find("ymax").text))
        shapes.append(
            {
                "label": name,
                "shape_type": "rectangle",
                "points": [[xmin, ymin], [xmax, ymax]],
            }
        )

    size = xmltree.find("size")
    imageWidth = int(float(size.find("width").text))
    imageHeight = int(float(size.find("height").text))
    return dict(shapes=shapes, imageWidth=imageWidth, imageHeight=imageHeight)


def make_imdb(img_dir, ann_dir, include=None):
    cache = []
    for img_path, ann_path in _filter(img_dir, ann_dir, include):
        if ann_path.suffix == ".xml":
            ann_data = xml2json(ann_path)
        elif ann_path.suffix == ".json":
            ann_data = load_json(ann_path)
        else:
            ann_data = dict(shapes=[], imageWidth=0, imageHeight=0)

        cache.append((img_path, ann_data, img_path.relative_to(img_dir)))
    return cache


def copyfile(out_dir, img_path, out_path, del_shapes):
    im = cv.imread(str(img_path), 1)

    for bbox in del_shapes:
        x, y, w, h = map(int, bbox)
        im[y: y + h, x: x + w] = 0

    cur_file = out_dir / "images" / out_path
    cur_file.parent.mkdir(parents=True, exist_ok=True)

    cv.imwrite(str(cur_file), im)
    return str(cur_file.relative_to(out_dir))


def make_dataset(img_dir, ann_dir=None, out_dir=None, include=None, mapping=None, min_size=0, all_imgs=True):
    imdb = make_imdb(img_dir, ann_dir, include)

    if out_dir is not None:
        out_dir = Path(out_dir)
    else:
        img_dir = Path(img_dir)
        out_dir = img_dir.name + "_coco"
        out_dir = img_dir.parent / out_dir
    shutil.rmtree(out_dir, ignore_errors=True)

    labels = set()
    for _, ann_data, _ in imdb:
        labels.update([s["label"] for s in ann_data["shapes"]])
    labels = sorted(labels)

    cat_index = {l: i for i, l in enumerate(labels)}

    imgs, anns = [], []
    img_id, ann_id = 0, 0
    for img_path, ann_data, out_path in imdb:
        shapes = ann_data["shapes"]
        img_w = ann_data["imageWidth"]
        img_h = ann_data["imageHeight"]

        if len(shapes) == 0:
            print(f"none-shapes: {img_path}")
            if not all_imgs:
                continue

        img_id += 1
        del_shapes = []
        for shape in shapes:
            label = shape["label"]
            points = shape["points"]
            shape_type = shape["shape_type"]

            if mapping is not None:
                label = _trans(mapping, label)

            if shape_type == "rectangle":
                assert len(points) == 2, "[[x1, y1], [x2, y2]]"
                xys = np.asarray(points)
                x_min, y_min = np.min(xys, axis=0)
                x_max, y_max = np.max(xys, axis=0)

                w, h = x_max - x_min, y_max - y_min
                bbox = _norm([x_min, y_min, w, h], img_w, img_h)
                x_mid, y_mid = (x_min + x_max) * 0.5, (y_min + y_max) * 0.5
                points = [(x_mid, y_min), (x_max, y_mid),
                          (x_mid, y_max), (x_min, y_mid)]
            elif shape_type == "polygon":
                assert len(points) >= 3, "[[x, y], [x, y], ...]"
                xys = np.asarray(points)
                x_min, y_min = np.min(xys, axis=0)
                x_max, y_max = np.max(xys, axis=0)

                w, h = x_max - x_min, y_max - y_min
                bbox = _norm([x_min, y_min, w, h], img_w, img_h)
            else:
                raise NotImplementedError(f"Not Implemented: {shape_type}")

            if min(bbox[2:]) < 3:
                continue

            w = max(min_size, w)
            h = max(min_size, h)

            if label in DEL_LABELS:
                del_shapes.append(bbox)
                continue

            ann_id += 1
            ann = dict(id=ann_id,
                       image_id=img_id,
                       category_id=cat_index[label],
                       segmentation=[np.asarray(points).flatten().tolist()],
                       area=np.prod(bbox[2:]), bbox=bbox, xyxy=nms.xywh2xyxy(bbox), iscrowd=0)
            anns.append(ann)

        img = dict(id=img_id,
                   width=img_w,
                   height=img_h,
                   file_name=copyfile(out_dir, img_path, out_path, del_shapes))
        imgs.append(img)

    cats = [dict(id=i, name=name, supercategory="")
            for i, name in enumerate(labels)]
    coco = dict(images=imgs, annotations=anns, categories=cats)
    print(f"imgs: {len(imgs)}, anns: {len(anns)}\nlabels: {labels}")
    save_json(coco, out_dir / "coco.json")
    return str(out_dir)
