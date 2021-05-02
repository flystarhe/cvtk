import numpy as np
from pathlib import Path
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

    return dts[np.argmax(rank1 * 100 + rank2 + rank3)]


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
