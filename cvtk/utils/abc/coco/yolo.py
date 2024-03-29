# https://github.com/ultralytics/JSON2YOLO
import shutil
from pathlib import Path

import numpy as np
from cvtk.io import load_json
from tqdm import tqdm


def yolo_from_coco(coco_dir, json_dir=None):
    coco_dir = Path(coco_dir)

    if json_dir is None:
        json_dir = coco_dir
    else:
        json_dir = coco_dir / json_dir

    out_dir = coco_dir.name + "_yolo"
    out_dir = coco_dir.parent / out_dir
    shutil.rmtree(out_dir, ignore_errors=True)

    (out_dir / "labels").mkdir(parents=True)
    (out_dir / "images").mkdir(parents=True)

    for json_file in sorted(json_dir.glob("*.json")):
        data = load_json(json_file)

        names = [c["supercategory"] + "." + c["name"]
                 for c in data["categories"]]

        images = {"%g" % x["id"]: x for x in data["images"]}
        cvt_id = {c["id"]: i for i, c in enumerate(data["categories"])}

        image_path_list = []
        for img in data["images"]:
            src_path = coco_dir / img["file_name"]
            dst_path = out_dir / "images" / src_path.name
            if not dst_path.exists():
                shutil.copyfile(src_path, dst_path)
            image_path_list.append(f"./images/{src_path.name}")
        image_path_list = sorted(set(image_path_list))
        n_images = len(image_path_list)

        with open(out_dir / "names.txt", "a") as file:
            file.write("{}: {}\n".format(json_file.stem, names))

        with open(out_dir / (json_file.stem + ".txt"), "w") as file:
            file.write("\n".join(image_path_list))

        pos = n_images // 5
        np.random.seed(pos * 100)
        np.random.shuffle(image_path_list)

        with open(out_dir / (json_file.stem + "_val.txt"), "w") as file:
            file.write("\n".join(image_path_list[:pos]))

        with open(out_dir / (json_file.stem + "_train.txt"), "w") as file:
            file.write("\n".join(image_path_list[pos:]))

        for x in tqdm(data["annotations"], desc="%s (%g)" % (json_file.stem, n_images)):
            if x.get("iscrowd"):
                continue

            img = images["%g" % x["image_id"]]
            h, w, f = img["height"], img["width"], img["file_name"]

            # format is [top left x, top left y, width, height]
            box = np.array(x["bbox"], dtype=np.float32)
            box[:2] += box[2:] / 2  # to center
            box[[0, 2]] /= w  # normalize x
            box[[1, 3]] /= h  # normalize y

            if (box[2] > 0.) and (box[3] > 0.):  # if w * h > 0
                with open(out_dir / "labels" / (Path(f).stem + ".txt"), "a") as file:
                    file.write("%g %.6f %.6f %.6f %.6f\n" %
                               (cvt_id[x["category_id"]], *box))
    return str(out_dir)
