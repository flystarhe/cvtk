import copy
import os
import re
from pathlib import Path

from cvtk.text.utils import replace2
from papermill import execute_notebook


def _gen_key(vals, carry_on):
    if carry_on == "latest":
        return "latest"

    if isinstance(carry_on, int):
        carry_on = [carry_on]

    assert isinstance(carry_on, list)

    return str([vals[i] for i in carry_on])


def ext_args(args, params):
    if args is None:
        return [[b] for b in params]
    return [a + [b] for a in args for b in params]


def run(carry_on, workdir, outdir, cfg, input_path_list, times_list, lr_list, gids):
    args = ext_args(None, input_path_list)
    args = ext_args(args, times_list)
    args = ext_args(args, lr_list)
    args = ext_args(args, gids)

    outdir = Path(outdir)

    logs = []
    _cached = {}
    for vals in args:
        _cfg = copy.deepcopy(cfg)
        input_path, times, lr, i = vals
        fname = f"{Path(input_path).stem}_{times=:02d}_{lr=:g}_{i=:d}.ipynb"

        _out = (outdir / fname[:-6]).as_posix()
        if carry_on is not None:
            key = _gen_key(vals, carry_on)
            _cfg["cfg_load_from"] = _cached.get(key)
            _cached[key] = os.path.join(_out, "latest.pth")
        _cfg = replace2(r".*coco_file$", _cfg, "/01/", f"/{i:02d}/")
        _cfg.update(cfg_experiment_path=_out, cfg_times=times, cfg_lr=lr)

        log = execute_notebook(input_path, outdir / fname, parameters=_cfg,
                               cwd=workdir, progress_bar=False,
                               kernel_name="python")
        logs.append(log)
    return logs


def clean_models(workdir, n=3):
    workdir = Path(workdir)
    pattern = re.compile(r'_(\d+)\.pth$')
    files = [str(f) for f in workdir.glob('*.pth')]

    def _num(name):
        try:
            return int(pattern.search(name).groups()[0])
        except Exception as e:
            print(f'Warning {name} - {e}')
        return 999

    files = sorted(files, key=lambda x: _num(x))[:-n]
    return files
