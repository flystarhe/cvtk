import shutil
import pandas as pd
from pathlib import Path


def _save_to(f, indir, outdir, level=None):
    if level is None:
        return outdir / f.name

    f = f.relative_to(indir)
    return outdir / Path(*f.parts[-level:])


def split_folder(indir, outdir, ref_file, col_path, col_label, suffixes=".jpg", level=None):
    indir = Path(indir)

    if outdir is None:
        outdir = f"{str(indir)}_TMP"

    outdir = Path(outdir)
    shutil.rmtree(outdir, ignore_errors=True)

    data = list(indir.glob("**/*.*"))
    data = {f.name: f for f in sorted(data)}

    src_list = list(data.values())

    if isinstance(suffixes, str):
        suffixes = set(suffixes.split(","))
        src_list = [f for f in src_list if f.suffix in suffixes]

    if ref_file is not None:
        if ref_file.endswith(".csv"):
            df = pd.read_csv(ref_file)
        elif ref_file.endswith(".xlsx"):
            df = pd.read_excel(ref_file, "Sheet1")
        else:
            raise Exception(f"{ref_file} not supported")

        mapping = {Path(row[col_path]).stem: row[col_label]
                   for _, row in df.iterrows()}
        dst_list = [_save_to(f, indir, outdir / mapping[f.stem])
                    for f in src_list if f.stem in mapping]
    else:
        dst_list = [_save_to(f, indir, outdir, level=level)
                    for f in src_list]

    for f in sorted(set([f.parent for f in dst_list])):
        f.mkdir(parents=True, exist_ok=True)

    for src, dst in zip(src_list, dst_list):
        shutil.copyfile(src, dst)

    return f"copy {len(src_list)} file to {outdir}"
