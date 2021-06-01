from pathlib import Path

import numpy as np
from cvtk.io import load_pkl, save_pkl
from cvtk.utils.abc.nms import clean_by_bbox
from scipy.stats import rankdata


def get_val(data, key, val=None):
    if key in data:
        return data[key]
    if "*" in data:
        return data["*"]
    return val


def gen_code_by_path(img_path, **kw):
    return Path(img_path).parent.name


def gen_code_by_complex(dts, **kw):
    if len(dts) == 0:
        return dict(bbox=[0, 0, 1, 1], xyxy=[0, 0, 1, 1], label="__OK", score=1.0)

    score_thr, label_grade = kw["score_thr"], kw["label_grade"]
    rank1 = np.array([dt["score"] >= get_val(score_thr, dt["label"], 0.3)
                      for dt in dts], dtype=np.int32)
    rank2 = np.array([get_val(label_grade, dt["label"], 1)
                      for dt in dts], dtype=np.int32)
    rank3 = np.array([dt["score"]
                      for dt in dts], dtype=np.float32)

    return dts[np.argmax(rank1 * 100 + rank2 * 10 + rank3)]


def gen_code_by_max_score(dts, **kw):
    if len(dts) == 0:
        return dict(bbox=[0, 0, 1, 1], xyxy=[0, 0, 1, 1], label="__OK", score=1.0)

    return max(dts, key=lambda x: x["score"])


def gen_code_by_rank_mixed(dts, **kw):
    if len(dts) == 0:
        return dict(bbox=[0, 0, 1, 1], xyxy=[0, 0, 1, 1], label="__OK", score=1.0)

    rank1 = rankdata([dt["bbox"][2] * dt["bbox"][3] for dt in dts])
    rank2 = rankdata([dt["score"] for dt in dts])
    return dts[np.argmax(rank1 + rank2)]


def image_label(x, **kw):
    """Use important defect as image label.

    Args:
        x (str, list): image path or list of dict.
   Returns:
        str(image label) or dict.
    """
    if isinstance(x, str):
        return gen_code_by_path(x)

    mode = kw.get("mode", "max_score")

    if mode == "complex":
        return gen_code_by_complex(x, **kw)
    elif mode == "max_score":
        return gen_code_by_max_score(x, **kw)
    elif mode == "rank_mixed":
        return gen_code_by_rank_mixed(x, **kw)
    else:
        raise NotImplementedError(f"Not Implemented mode={mode}")


def gen_test(results, mode=None, score_thr=None, label_grade=None, **kw):
    """Show model prediction results, allow gts is empty.

    Args:
        results (list): List of `tuple(img_path, target, predict, dts, gts)`
        mode (str): Optional value in `{complex, max_score, rank_mixed}`
        score_thr (dict): Such as `{"CODE1":S1, "CODE2":S2, "*":0.3}`
        label_grade (dict): Such as `{"CODE1":L1, "CODE2":L2, "*":1}`
        clean_mode (str): Optional value in `{dist, iou, min, ...}`
    """
    if score_thr is None:
        score_thr = {"*": 0.3}

    if label_grade is None:
        label_grade = {"*": 1}

    clean_mode = kw.pop("clean_mode", None)
    clean_param = kw.pop("clean_param", None)

    temp_file = Path("tmp/test.pkl")
    if isinstance(results, str):
        temp_file = Path(results)
        results = load_pkl(results)
    temp_file = temp_file.with_suffix(".zzz")
    temp_file.parent.mkdir(parents=True, exist_ok=True)

    outputs = []
    for file_name, target, predict, dts, gts in results:
        if clean_mode is not None:
            dts = clean_by_bbox(dts, clean_mode, clean_param)
        if mode is not None:
            predict = image_label(
                dts, mode=mode, score_thr=score_thr, label_grade=label_grade, **kw)
        outputs.append([file_name, target, predict, dts, gts])

    return save_pkl(outputs, str(temp_file))
