from collections import defaultdict
from pathlib import Path

import albumentations as A
import cv2 as cv
import numpy as np
from cvtk.io import load_json
from torchvision.transforms import functional as F


def check_bboxes(src_bboxes, dst_bboxes):
    src_w = src_bboxes[:, 2] - src_bboxes[:, 0]
    src_h = src_bboxes[:, 3] - src_bboxes[:, 1]
    src_area = src_w * src_h

    dst_w = dst_bboxes[:, 2] - dst_bboxes[:, 0]
    dst_h = dst_bboxes[:, 3] - dst_bboxes[:, 1]
    dst_area = dst_w * dst_h

    s1 = (dst_area >= src_area * 0.5)
    s2 = (dst_w >= src_w) * (dst_h >= src_w * 1.5)
    s3 = (dst_h >= src_h) * (dst_w >= src_h * 1.5)
    ss = s1 + s2 + s3

    inner = (dst_w >= 4) * (dst_h >= 4)
    return ss * inner, np.logical_not(ss) * inner


class ToyDataset:

    def __init__(self, data_root, coco_file, single_cls=True, crop_size=480, phase="train"):
        self._ann_file = Path(data_root) / coco_file
        self._img_prefix = Path(data_root)
        self._single_cls = single_cls
        self._crop_size = crop_size
        self._phase = phase

        if phase == "train":
            self.transforms = A.Compose([
                A.SmallestMaxSize(max_size=512, interpolation=cv.INTER_LINEAR, p=1.0),
                A.RandomBrightnessContrast(p=0.2),
                A.Flip(p=0.5),
            ], bbox_params=A.BboxParams(format="pascal_voc", label_fields=["labels"]))
        else:
            self.transforms = A.Compose([
                A.SmallestMaxSize(max_size=512, interpolation=cv.INTER_LINEAR, p=1.0),
            ], bbox_params=A.BboxParams(format="pascal_voc", label_fields=["labels"]))

        self.mean = [0.485, 0.456, 0.406]
        self.std = [0.229, 0.224, 0.225]
        self.create_index()

    def create_index(self):
        coco = load_json(self._ann_file)

        cats = coco["categories"]
        if self._single_cls:
            names = ["_BG", "_FG"]
            id2label = {c["id"]: 1 for c in cats}
        else:
            names = ["_BG"] + [c["name"] for c in cats]
            id2label = {c["id"]: i for i, c in enumerate(cats, 1)}
        imgs = {img["id"]: img["file_name"] for img in coco["images"]}

        imdb = defaultdict(list)
        for ann in coco["annotations"]:
            x, y, w, h = ann["bbox"]
            bbox = [x, y, x + w, y + h]
            label = id2label[ann["category_id"]]
            imdb[ann["image_id"]].append((bbox, label))

        ids = set(imgs.keys()) & set(imdb.keys())

        self.imgs = imgs
        self.imdb = imdb
        self.names = names
        self.ids = sorted(ids)

    def random_crop(self, image, bboxes, labels):
        img_h, img_w = image.shape[:2]

        if self._crop_size < min(img_h, img_w):
            bboxes = np.asarray(bboxes, dtype=np.int64)
            labels = np.asarray(labels, dtype=np.int64)
            old_bboxes = bboxes.copy()

            n = bboxes.shape[0]
            ind = np.random.randint(n)
            x1, y1, x2, y2 = bboxes[ind].tolist()

            if x2 - x1 < self._crop_size:
                x1 = max(0, np.random.randint(x2 - self._crop_size, x1))
            if y2 - y1 < self._crop_size:
                y1 = max(0, np.random.randint(y2 - self._crop_size, y1))

            x2, y2 = x1 + self._crop_size, y1 + self._crop_size

            if x2 > img_w:
                offset = x2 - img_w
                x1, x2 = x1 - offset, x2 - offset

            if y2 > img_h:
                offset = y2 - img_h
                y1, y2 = y1 - offset, y2 - offset

            image = image[y1: y2, x1: x2]

            bboxes[:, 0::2] = np.clip(bboxes[:, 0::2], x1, x2)
            bboxes[:, 1::2] = np.clip(bboxes[:, 1::2], y1, y2)
            bboxes -= np.asarray([x1, y1, x1, y1])

            dst_mask, drop_mask = check_bboxes(old_bboxes, bboxes)

            for x1, y1, x2, y2 in bboxes[drop_mask].tolist():
                image[y1: y2, x1: x2] = 0

            bboxes = bboxes[dst_mask].tolist()
            labels = labels[dst_mask].tolist()
            return image, bboxes, labels

        return image, bboxes, labels

    def __getitem__(self, index):
        image_id = self.ids[index]

        file_name = self.imgs[image_id]
        image_path = str(self._img_prefix / file_name)

        image = cv.imread(image_path, 1)
        image = cv.cvtColor(image, cv.COLOR_BGR2RGB)
        bboxes, labels = map(list, zip(*self.imdb[image_id]))

        if self._phase == "train":
            image, bboxes, labels = self.random_crop(image, bboxes, labels)

        augmented = self.transforms(image=image, bboxes=bboxes, labels=labels)
        image, bboxes, labels = augmented["image"], augmented["bboxes"], augmented["labels"]

        hw = image.shape[:2]
        image = F.to_tensor(image)  # \in [0, 1]
        image = F.normalize(image, self.mean, self.std, True)
        target = dict(id=image_id, bboxes=bboxes, labels=labels, img_shape=hw)
        return image, target

    def __len__(self):
        return len(self.ids)


def get_dataset(data_root, coco_file, single_cls=True, crop_size=480, phase="train"):
    dataset = ToyDataset(data_root, coco_file, single_cls, crop_size, phase)
    return dataset, len(dataset.names)
