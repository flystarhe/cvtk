import argparse
import multiprocessing
import os
import os.path as osp
import subprocess
import time
from pathlib import Path
from collections import defaultdict
from cvtk.io import load_json, load_pkl, save_json, save_pkl
from cvtk.utils.abc.gen import image_label
from scipy.stats import rankdata

IMG_EXTENSIONS = set([".jpg", ".jpeg", ".png", ".bmp"])
G_COMMAND = "CUDA_VISIBLE_DEVICES={} python dev/py_app.py {} {} {} -b {}"


def system_command(params):
    gpu_id, file_name, config, checkpoint, batch_size = params
    command_line = G_COMMAND.format(
        gpu_id, file_name, config, checkpoint, batch_size)
    result = subprocess.run(command_line, shell=True, stdout=subprocess.PIPE)
    if result.returncode != 0:
        return "GPU{} {}: {}".format(gpu_id, file_name, result.args)
    return "GPU{} {}: {}".format(gpu_id, file_name, "OK.")


def task_split(dataset, splits, tmp_dir):
    n = len(dataset)
    block_size = (n - 1) // splits + 1

    file_list = []
    os.makedirs(tmp_dir, exist_ok=True)
    for i, i_start in enumerate(range(0, n, block_size)):
        subset = dataset[i_start:i_start + block_size]
        out_file = osp.join(tmp_dir, f"part{i:02d}")
        save_json(subset, out_file)
        file_list.append(out_file)
    return file_list


def collect_results(pkl_list):
    results = []
    for pkl_file in pkl_list:
        try:
            data = load_pkl(pkl_file)
            results.extend(data)
        except:
            print("Failed: {}".format(pkl_file))
    return results


def multi_gpu_test(dataset, gpus, config, checkpoint, batch_size=1, workers_per_gpu=2):
    file_list = task_split(dataset, gpus * workers_per_gpu, "tmp/")
    params = [(i % gpus, f, config, checkpoint, batch_size)
              for i, f in enumerate(file_list)]

    pool = multiprocessing.Pool(processes=gpus * workers_per_gpu)
    logs = pool.map(system_command, params)
    print("\n".join(logs))

    pkl_list = [f + ".pkl" for f in file_list]
    return collect_results(pkl_list)


def test_coco(data_root, coco_file, gpus, config, checkpoint, batch_size=1, workers_per_gpu=2):
    if coco_file == "none":
        imgs = [str(img) for img in Path(data_root).glob("**/*")
                if img.suffix in IMG_EXTENSIONS]
        gts = defaultdict(list)
    else:
        coco_file = osp.join(data_root, coco_file)
        coco = load_json(coco_file)

        id2label = {c["id"]: c["name"] for c in coco["categories"]}

        im2gts = defaultdict(list)
        for a in coco["annotations"]:
            a["label"] = id2label[a["category_id"]]
            a["score"] = 1.0
            im2gts[a["image_id"]].append(a)

        imgs = []
        gts = defaultdict(list)
        for img in coco["images"]:
            img_path = osp.join(data_root, img["file_name"])
            gts[img_path] = im2gts[img["id"]]
            imgs.append(img_path)

    results = multi_gpu_test(
        imgs, gpus, config, checkpoint, batch_size, workers_per_gpu)
    assert len(imgs) == len(results)

    outputs = []
    for img_path, dts in results:
        target = image_label(img_path)
        predict = image_label(dts, mode="max_score")
        outputs.append([img_path, target, predict, dts, gts[img_path]])
    temp_file = "test_{}.pkl".format(time.strftime("%m%d%H%M"))
    temp_file = osp.join(osp.dirname(checkpoint), temp_file)
    return save_pkl(outputs, temp_file)


def main(args):
    kw = vars(args)
    return test_coco(**kw)


if __name__ == "__main__":
    print("\n{:#^64}\n".format(__file__))
    parser = argparse.ArgumentParser(description=None)
    parser.add_argument("data_root", type=str,
                        help="dataset root")
    parser.add_argument("coco_file", type=str,
                        help="coco json file")
    parser.add_argument("gpus", type=int,
                        help="number of gpus")
    parser.add_argument("config", type=str,
                        help="config file path")
    parser.add_argument("checkpoint", type=str,
                        help="checkpoint file path")
    parser.add_argument("batch_size", type=int, default=1,
                        help="batch size")
    parser.add_argument("workers_per_gpu", type=int, default=2,
                        help="workers per gpu")
    args = parser.parse_args()
    print(main(args))
