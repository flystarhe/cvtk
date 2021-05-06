import glob
import pickle
import re
import shutil

try:
    import simplejson as json
except ImportError:
    import json

from pathlib import Path


def make_dir(path):
    path = Path(path)
    if not path.exists():
        path.mkdir(parents=True)
    return path


def increment_path(path, exist_ok=True, sep=""):
    path = Path(path)

    if not path.exists() or exist_ok:
        return make_dir(path)

    dirs = glob.glob(f"{path}{sep}*")
    pattern = re.compile(rf"{path.name}{sep}(\d+)")

    matches = [pattern.search(d) for d in dirs]
    i = [int(m.groups()[0]) for m in matches if m]

    n = max(i) + 1 if i else 2
    return make_dir(f"{path}{sep}{n}")


def copyfile(src, dst):
    # copies the file src to the file or directory dst
    return shutil.copy(src, dst)


def copyfile2(src, dst):
    # dst must be the complete target file name
    return shutil.copyfile(src, dst)


def load_pkl(pkl_file):
    with open(pkl_file, "rb") as f:
        data = pickle.load(f)
    return data


def save_pkl(data, pkl_file):
    make_dir(Path(pkl_file).parent)
    with open(pkl_file, "wb") as f:
        pickle.dump(data, f)
    return pkl_file


def load_json(json_file):
    with open(json_file, "r") as f:
        data = json.load(f)
    return data


def save_json(data, json_file):
    make_dir(Path(json_file).parent)
    with open(json_file, "w") as f:
        json.dump(data, f, indent=4)
    return json_file


def load_csv(csv_file):
    with open(csv_file, "r") as f:
        lines = f.readlines()
    return lines


def save_csv(lines, csv_file):
    make_dir(Path(csv_file).parent)
    with open(csv_file, "w") as f:
        f.write("\n".join(lines))
    return csv_file
