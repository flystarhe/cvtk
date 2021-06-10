import copy
import re
from pathlib import Path

from papermill import execute_notebook


def _update(cfg, group_id):
    pattern = re.compile(r".*coco_file$", flags=0)

    old, new = "/01/", f"/{group_id:02d}/"

    def _func(k, v):
        if pattern.match(k):
            if isinstance(v, str):
                return v.replace(old, new)
        return v

    return {k: _func(k, v) for k, v in cfg.items()}


def _gen_key(data, carry_on):
    if carry_on == "latest":
        return "latest"

    if isinstance(carry_on, str):
        carry_on = [carry_on]

    assert isinstance(carry_on, list)

    return ",".join([f"{k}={data[k]}" for k in sorted(carry_on)])


def ext_args(args, params):
    if args is None:
        return [[b] for b in params]
    return [a + [b] for a in args for b in params]


def run(task_list, carry_on, workdir, outdir, cfg):
    # task_list = [dict(cfg_times=2, cfg_lr=0.02, group_id=1, input_path="")]
    outdir = Path(outdir)
    logs, _cached = [], {}
    for num, data in enumerate(task_list, 1):
        _cfg = copy.deepcopy(cfg)

        input_path = None
        for name, val in data.items():
            if name == "input_path":
                input_path = Path(val)
            elif name == "group_id":
                _cfg = _update(_cfg, val)
            elif name in _cfg:
                _cfg[name] = val
        assert input_path is not None

        this_dir = outdir / f"{num:02d}"
        this_dir.mkdir(parents=True, exist_ok=True)

        if carry_on is not None:
            key = _gen_key(data, carry_on)
            _cfg["cfg_load_from"] = _cached.get(key)
            _cached[key] = str(this_dir / "latest.pth")
            _cached["latest"] = str(this_dir / "latest.pth")
        _cfg["cfg_experiment_path"] = str(this_dir)

        output_path = this_dir / input_path.name
        log = execute_notebook(input_path, output_path, parameters=_cfg,
                               cwd=workdir, progress_bar=False,
                               kernel_name="python")
        logs.append(log)
    return logs, _cached


def clean_models(workdir, n=3):
    workdir = Path(workdir)
    pattern = re.compile(r"_(\d+)\.pth$")
    files = [str(f) for f in workdir.glob("*.pth")]

    def _num(name):
        try:
            return int(pattern.search(name).groups()[0])
        except Exception as e:
            print(f"Warning {name} - {e}")
        return 999

    files = sorted(files, key=lambda x: _num(x))[:-n]
    return files
