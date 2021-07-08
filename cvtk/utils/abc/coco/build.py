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
        is_in, include = True, str(include)
        if include.startswith("-"):
            is_in, include = False, include[1:]
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
        if is_in:
            img_list = [x for x in img_list if x.stem in targets]
        else:
            img_list = [x for x in img_list if x.stem not in targets]

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
    imdb = []
    label_files = []
    for img_path, ann_path in _filter(img_dir, ann_dir, include):
        try:
            if ann_path.suffix == ".xml":
                label_files.append(ann_path)
                ann_data = xml2json(ann_path)
            elif ann_path.suffix == ".json":
                label_files.append(ann_path)
                ann_data = load_json(ann_path)
            elif ann_path.suffix in IMG_EXTENSIONS:
                ann_data = dict(shapes=[], imageWidth=0, imageHeight=0)
            else:
                assert False, f"{ann_path.suffix} not supported"

            imdb.append((ann_data, img_path, img_path.relative_to(img_dir)))
        except Exception as e:
            print(f"{ann_path.name} - {e}")
    return imdb, label_files


def copyfile(nparr, out_dir, out_path, del_shapes):
    for bbox in del_shapes:
        x, y, w, h = map(int, bbox)
        cv.rectangle(nparr, (x, y), (x + w, y + h), (0, 0, 255), -1)

    cur_file = out_dir / "images" / out_path
    cur_file.parent.mkdir(parents=True, exist_ok=True)

    cv.imwrite(str(cur_file), nparr)
    return str(cur_file.relative_to(out_dir))


def make_dataset(img_dir, ann_dir=None, out_dir=None, include=None, mapping=None, min_size=0, all_imgs=True):
    imdb, label_files = make_imdb(img_dir, ann_dir, include)

    if out_dir is not None:
        out_dir = Path(out_dir)
    else:
        img_dir = Path(img_dir)
        out_dir = img_dir.name + "_coco"
        out_dir = img_dir.parent / out_dir
    shutil.rmtree(out_dir, ignore_errors=True)
    out_dir.mkdir(parents=True, exist_ok=True)

    labels = set()
    for ann_data, _,  _ in imdb:
        labels.update([s["label"] for s in ann_data["shapes"]])

    if mapping is not None:
        labels = set([_trans(mapping, l) for l in labels])

    if not labels:
        labels = set(["_FG"])

    labels = sorted(labels - DEL_LABELS)
    print(f"\nlabels: {len(labels)}\n{labels}\n")
    cat_index = {l: i for i, l in enumerate(labels)}

    imgs, anns = [], []
    img_id, ann_id = 0, 0
    for ann_data, img_path, out_path in imdb:
        shapes = ann_data["shapes"]
        img_w = ann_data["imageWidth"]
        img_h = ann_data["imageHeight"]

        try:
            nparr = cv.imread(str(img_path), 1)
            assert min(nparr.shape[:2]) > 32, "image size is too small"
            if img_w + img_h == 0:
                img_h, img_w = nparr.shape[:2]
            assert (img_h, img_w) == nparr.shape[:2], "image size not equal"
        except Exception as e:
            print(f"bad-image: {img_path}\n  {e}")
            continue

        if (len(shapes) == 0) and (not all_imgs):
            print(f"skip-image: {img_path}")
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

                x_mid, y_mid = (x_min + x_max) * 0.5, (y_min + y_max) * 0.5
                points = [(x_mid, y_min), (x_max, y_mid),
                          (x_mid, y_max), (x_min, y_mid)]
            elif shape_type == "polygon":
                assert len(points) >= 3, "[[x, y], [x, y], ...]"
                xys = np.asarray(points)
                x_min, y_min = np.min(xys, axis=0)
                x_max, y_max = np.max(xys, axis=0)

                w, h = x_max - x_min, y_max - y_min
            else:
                raise NotImplementedError(f"Not Implemented: {shape_type}")

            x_pad = (min_size - w + 1) // 2
            if x_pad > 0:
                x_min = x_min - x_pad
                w = w + x_pad * 2

            y_pad = (min_size - h + 1) // 2
            if y_pad > 0:
                y_min = y_min - y_pad
                h = h + y_pad * 2

            bbox = _norm([x_min, y_min, w, h], img_w, img_h)

            if label in DEL_LABELS:
                del_shapes.append(bbox)
                continue

            ann_id += 1
            ann = dict(id=ann_id,
                       image_id=img_id,
                       category_id=cat_index[label],
                       segmentation=[np.asarray(points).flatten().tolist()],
                       area=bbox[2] * bbox[3], bbox=bbox, xyxy=nms.xywh2xyxy(bbox), iscrowd=0)
            anns.append(ann)

        img = dict(id=img_id,
                   width=img_w,
                   height=img_h,
                   file_name=copyfile(nparr, out_dir, out_path, del_shapes))
        imgs.append(img)

    bak_dir = out_dir / "labels"
    bak_dir.mkdir(parents=True, exist_ok=True)
    dat = set([Path(img["file_name"]).stem for img in imgs])
    for label_file in label_files:
        if Path(label_file).stem in dat:
            shutil.copy(label_file, bak_dir)

    cats = [dict(id=i, name=name, supercategory="")
            for i, name in enumerate(labels)]
    coco = dict(images=imgs, annotations=anns, categories=cats)
    print(f"imgs: {len(imgs)}, anns: {len(anns)}, cats: {len(cats)}")
    save_json(coco, out_dir / "coco.json")
    return str(out_dir)
