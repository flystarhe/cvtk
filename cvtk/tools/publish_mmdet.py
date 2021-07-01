import re
import shutil
import subprocess
import sys
from argparse import ArgumentParser
from pathlib import Path

import torch


def list_models(workdir, suffix=".pth"):
    workdir = Path(workdir)

    if suffix.startswith("."):
        suffix = suffix[1:]

    pattern = re.compile(rf"_(\d+)\.{suffix}$")
    files = [str(f) for f in workdir.glob(f"*.{suffix}")]

    def _num(name):
        try:
            return int(pattern.search(name).groups()[0])
        except Exception:
            print(f"skip: {name}")
        return -1

    files = sorted(files, key=lambda x: _num(x))
    return files


def other_files(workdir, suffixes):
    workdir = Path(workdir)

    suffixes = set(suffixes)
    files = [str(f) for f in workdir.glob("*.*") if f.suffix in suffixes]

    return files


def parse_args(args):
    parser = ArgumentParser(description="publish mmdet")
    parser.add_argument("indir", type=str,
                        help="input dir")
    parser.add_argument("-o", "--outdir", type=str, default=None,
                        help="output dir")
    parser.add_argument("-f", "--suffixes", type=str, default=None,
                        help="1st is model, default `.pth,.py,.html`")
    args = parser.parse_args(args=args)

    kw = vars(args)
    return kw


def process_checkpoint(indir, outdir=None, suffixes=None):
    indir = Path(indir)

    if outdir is None:
        outdir = f"{str(indir)}_PUB"

    outdir = Path(outdir)
    shutil.rmtree(outdir, ignore_errors=True)
    outdir.mkdir(parents=True, exist_ok=True)

    if suffixes is None:
        suffixes = ".pth,.py,.html".split(",")
    else:
        suffixes = suffixes.split(",")

    checkpoints = list_models(indir, suffix=suffixes[0])
    checkpoint = torch.load(checkpoints[-1], map_location="cpu")

    if "optimizer" in checkpoint:
        del checkpoint["optimizer"]

    tmp_file = outdir / "model.pth"
    torch.save(checkpoint, str(tmp_file))

    sha = subprocess.check_output(["sha256sum", str(tmp_file)]).decode()
    final_file = outdir / f"model-{sha[:8]}.pth"

    subprocess.Popen(["mv", str(tmp_file), str(final_file)])

    dst = str(outdir)
    for src in other_files(indir, suffixes[1:]):
        shutil.copy(src, dst)

    return "\n".join([str(f) for f in outdir.glob("*.*")])


def main(args):
    kw = parse_args(args)
    print(f"\nkwargs: {kw}\n")
    return process_checkpoint(**kw)


if __name__ == "__main__":
    args = sys.argv[1:]
    print(main(args))
    sys.exit(0)
