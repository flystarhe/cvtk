import argparse
import cv2 as cv
import numpy as np
import torch
from cvtk.io import load_json, save_pkl
from mmcv.parallel import collate
from mmcv.parallel import scatter
from mmdet.apis import init_detector
from mmdet.datasets.pipelines import Compose


################################################################
from mmdet.datasets.builder import DATASETS
from mmdet.datasets.builder import PIPELINES
from py_abc import CocoDataset, RandomCrop, Resize
DATASETS.register_module(name="CocoDataset", force=True, module=CocoDataset)
PIPELINES.register_module(name="RandomCrop", force=True, module=RandomCrop)
PIPELINES.register_module(name="Resize", force=True, module=Resize)
################################################################


def json_encode(_result, _classes):
    if isinstance(_result, tuple):
        _result = _result[0]

    dts = []
    for i, bboxes in enumerate(_result):
        label = _classes[i]
        for tl_x, tl_y, br_x, br_y, score in bboxes.tolist():
            dt = dict(
                bbox=[tl_x, tl_y, br_x - tl_x, br_y - tl_y],
                xyxy=[tl_x, tl_y, br_x, br_y],
                label=label, score=score,
            )
            dts.append(dt)
    return dts


def inference_detector(imgs, model, device=None, test_pipeline=None):
    # imgs (list[ndarray]): `[cv.imread(file_name, 1), ...]`
    classes = model.CLASSES

    if device is None:
        device = next(model.parameters()).device

    if test_pipeline is None:
        cfg = model.cfg.copy()
        test_pipeline = cfg.data.test.pipeline
        test_pipeline[0].type = "LoadImageFromWebcam"
        test_pipeline = Compose(test_pipeline)

    data = [test_pipeline(dict(img=img)) for img in imgs]
    data = collate(data, samples_per_gpu=len(data))

    # just get the actual data from DataContainer
    data["img_metas"] = [img_metas.data[0] for img_metas in data["img_metas"]]
    data["img"] = [img.data[0] for img in data["img"]]
    data = scatter(data, [device])[0]
    with torch.no_grad():
        results = model(return_loss=False, rescale=True, **data)
    return [json_encode(r, classes) for r in results]


def test_imgs(imgs, config, checkpoint, batch_size=1):
    model = init_detector(config, checkpoint, device="cuda:0")
    device = next(model.parameters()).device

    cfg = model.cfg.copy()
    test_pipeline = cfg.data.test.pipeline
    test_pipeline[0].type = "LoadImageFromWebcam"
    test_pipeline = Compose(test_pipeline)

    results = []
    for i in range(0, len(imgs), batch_size):
        nparr_list = [cv.imread(f, 1) for f in imgs[i:i + batch_size]]
        results.extend(inference_detector(
            nparr_list, model, device, test_pipeline))
    return list(zip(imgs, results))


def main(args):
    in_file = args.data
    imgs = load_json(in_file)
    outputs = test_imgs(imgs, args.config,
                        args.checkpoint, args.batch_size)
    return save_pkl(outputs, in_file + ".pkl")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=None)
    parser.add_argument("data", type=str,
                        help="json file path")
    parser.add_argument("config", type=str,
                        help="config file path")
    parser.add_argument("checkpoint", type=str,
                        help="checkpoint file path")
    parser.add_argument("-b", "--batch_size", type=int, default=1,
                        help="batch size")
    args = parser.parse_args()
    print(main(args))
