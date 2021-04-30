from cvtk.io import load_json
from cvtk.transforms.mmdet import RandomCrop, Resize
from mmdet.datasets import CocoDataset as _CocoDataset


class CocoDataset(_CocoDataset):

    def load_annotations(self, ann_file):
        cats = load_json(ann_file)["categories"]
        self.CLASSES = [cat["name"] for cat in cats]
        data_infos = super(CocoDataset, self).load_annotations(ann_file)
        return data_infos
