import copy
import numpy as np
import shutil
from collections import defaultdict
from itertools import combinations
from pathlib import Path

from cvtk.io import load_json
from cvtk.io import make_dir
from cvtk.io import save_json


def save_dataset(coco, out_file, image_ids=None):
    coco = copy.deepcopy(coco)

    if image_ids is not None:
        image_ids = set(image_ids)
        coco["images"] = [img for img in coco["images"] if img["id"] in image_ids]
        coco["annotations"] = [a for a in coco["annotations"] if a["image_id"] in image_ids]

    print(f"{out_file} images: {len(coco['images'])}")
    return save_json(coco, out_file)


class KeepPSamplesIn(object):

    def __init__(self, p):
        """Keep P Sample(s) In task

        Args:
            p (int, float): Number of samples (``p``) to keep
        """
        assert isinstance(p, (int, float)) and (p > 0)
        self.p = p

    def split(self, coco_file, stratified=True, seed=1000):
        coco = load_json(coco_file)
        out_dir = Path(coco_file).parent
        out_dir = out_dir / "keep_p_samples"
        shutil.rmtree(out_dir, ignore_errors=True)

        cache = defaultdict(set)
        labels = defaultdict(set)
        for a in coco["annotations"]:
            cache[a["category_id"]].add(a["image_id"])
            labels[a["image_id"]].add(a["category_id"])

        if stratified:
            groups = defaultdict(list)
            for img in coco["images"]:
                ranks, image_id = [], img["id"]
                for category_id in labels[image_id]:
                    ranks.append((category_id, len(cache[category_id])))
                groups[self._get_group(ranks)].append(image_id)
        else:
            groups = {"none": [img["id"] for img in coco["images"]]}

        print([(k, len(groups[k])) for k in sorted(groups.keys())])
        for i in range(1, 11):
            test_index, train_index = self._split(groups, seed * i)

            this_dir = make_dir(out_dir / f"{i}")
            save_dataset(coco, this_dir / "all.json", None)
            save_dataset(coco, this_dir / "test.json", test_index)
            save_dataset(coco, this_dir / "train.json", train_index)
        return str(out_dir)

    def _get_group(self, ranks):
        c = 1000
        if ranks:
            c = min(ranks, key=lambda x: x[1] + x[0] / 1000)[0]
        return f"{c:04d}"

    def _split(self, groups, seed):
        test_index, train_index = [], []

        for _, image_ids in groups.items():
            image_ids = sorted(image_ids)

            p = self.p
            if isinstance(p, float):
                p = 1 + int(p * len(image_ids))

            np.random.seed(seed)
            np.random.shuffle(image_ids)
            if p >= len(image_ids):
                test_index.extend(image_ids)
                train_index.extend(image_ids)
            else:
                test_index.extend(image_ids[p:])
                train_index.extend(image_ids[:p])
        return test_index, train_index


class LeavePGroupsOut(object):

    def __init__(self, p_groups, limit=1000):
        """Leave P Group(s) Out cross-validator

        Args:
            p_groups (int): Number of groups (``p``) to leave out
        """
        self.p_groups = p_groups
        self.limit = limit

    def split(self, coco_file, seed=1000):
        coco = load_json(coco_file)
        out_dir = Path(coco_file).parent
        out_dir = out_dir / "leave_p_groups"
        shutil.rmtree(out_dir, ignore_errors=True)

        flags = np.zeros((len(coco["images"]), len(coco["categories"])), dtype=np.int64)
        cat_index = {cat["id"]: i for i, cat in enumerate(coco["categories"])}
        img_index = {img["id"]: i for i, img in enumerate(coco["images"])}
        assert flags.shape[0] == len(img_index)
        for a in coco["annotations"]:
            col = cat_index[a["category_id"]]
            row = img_index[a["image_id"]]
            flags[row, col] = 1

        for i, test_index in enumerate(self._iter_test_masks(flags), 1):
            self._split(coco, out_dir / f"{i}", test_index, seed * i)
        return str(out_dir)

    def _iter_test_masks(self, flags):
        groups = flags.sum(axis=0).nonzero()[0]

        n_groups = len(groups)
        if self.p_groups >= n_groups:
            raise ValueError(f"p_groups({self.p_groups}) not less than unique_groups({n_groups})")

        for indices in combinations(groups, self.p_groups):
            yield flags[:, indices].sum(axis=1) >= 1

    def _split(self, coco, this_dir, test_index, seed):
        this_dir = make_dir(this_dir)
        train_index = np.logical_not(test_index)
        image_ids = np.asarray([img["id"] for img in coco["images"]])

        test_index = image_ids[test_index]
        train_index = image_ids[train_index]

        if len(train_index) > self.limit:
            np.random.seed(seed)
            np.random.shuffle(train_index)
            train_index = train_index[:self.limit]

        save_dataset(coco, this_dir / "all.json", None)
        save_dataset(coco, this_dir / "test.json", test_index)
        save_dataset(coco, this_dir / "train.json", train_index)
