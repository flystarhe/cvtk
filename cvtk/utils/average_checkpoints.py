# https://github.com/pytorch/vision/blob/master/references/classification/utils.py#L259
import re
import sys
from collections import OrderedDict
from pathlib import Path

import torch
from torch.serialization import default_restore_location


def average_checkpoints(inputs):
    """Loads checkpoints from inputs and returns a model with averaged weights.

    Args:
      inputs (List[str]): An iterable of string paths of checkpoints to load from.
    Returns:
      A dict of string keys mapping to various values. The 'model' key
      from the returned dict should correspond to an OrderedDict mapping
      string parameter names to torch Tensors.
    """
    params_dict = OrderedDict()
    params_keys = None
    new_state = None
    num_models = len(inputs)
    for fpath in inputs:
        with open(fpath, "rb") as f:
            state = torch.load(
                f,
                map_location=(
                    lambda s, _: default_restore_location(s, "cpu")
                ),
            )
        # Copies over the settings from the first checkpoint
        if new_state is None:
            new_state = state
        model_params = state["model"]
        model_params_keys = list(model_params.keys())
        if params_keys is None:
            params_keys = model_params_keys
        elif params_keys != model_params_keys:
            raise KeyError(
                "For checkpoint {}, expected list of params: {}, "
                "but found: {}".format(f, params_keys, model_params_keys)
            )
        for k in params_keys:
            p = model_params[k]
            if isinstance(p, torch.HalfTensor):
                p = p.float()
            if k not in params_dict:
                params_dict[k] = p.clone()
                # NOTE: clone() is needed in case of p is a shared parameter
            else:
                params_dict[k] += p
    averaged_params = OrderedDict()
    for k, v in params_dict.items():
        averaged_params[k] = v
        if averaged_params[k].is_floating_point():
            averaged_params[k].div_(num_models)
        else:
            averaged_params[k] //= num_models
    new_state["model"] = averaged_params
    return new_state


def last_n(root, n=2, m=1):
    pattern = re.compile(r"_(\d+)")
    files = Path(root).glob("model_*")

    def _num(name):
        try:
            return int(pattern.search(str(name)).group(1))
        except Exception as e:
            print(f"Warning {name} - {e}")
        return -1

    files = sorted(files, key=lambda x: _num(x), reverse=True)
    return files[::m][:n]


def avg(paths, f="model_avg.pth"):
    weights = average_checkpoints(paths)
    torch.save(weights, f)
    return f


if __name__ == "__main__":
    argv = sys.argv[1:]
    assert len(argv) == 2
    argv = [int(x) for x in argv]

    paths = last_n(".", *argv)
    sys.exit(avg(paths))
