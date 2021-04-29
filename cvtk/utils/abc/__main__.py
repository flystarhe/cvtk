import os
import sys
from argparse import ArgumentParser

from cvtk.utils.abc.coco.build import make_dataset as coco_build
from cvtk.utils.abc.coco.yolo import yolo_from_coco as coco_to_yolo
from cvtk.utils.abc.coco.sampling import KeepPSamplesIn, LeavePGroupsOut
from cvtk.utils.abc.patch import patch_images
from cvtk.utils.abc.visualize import display_coco, display_test

# Remove "" and current working directory from the first entry of sys.path
while True:
    if sys.path[0] in ("", os.getcwd()):
        sys.path.pop(0)
    else:
        break


def args_coco_build(args=None):
    parser = ArgumentParser(description="build dataset")
    parser.add_argument("img_dir", type=str,
                        help="images dir")
    parser.add_argument("-a", "--ann_dir", type=str, default=None,
                        help="labels dir, search annotation from here")
    parser.add_argument("-o", "--out_dir", type=str, default=None,
                        help="output dir, default save to `{img_dir}_coco`")
    parser.add_argument("-i", "--include", type=str, default=None,
                        help="filter dataset, hiplot(*.csv), coco(*.json) or dir(path/)")
    parser.add_argument("-m", "--mapping", type=str, default=None,
                        help="python dict, be run `eval(mapping)`")
    parser.add_argument("-e", "--min_size", type=int, default=0,
                        help="expand mini box to `min_size`")
    parser.add_argument("--all-imgs", action="store_true",
                        help="keep none-shapes image")
    args = parser.parse_args(args=args)

    kw = vars(args)
    mapping = kw.pop("mapping")
    if mapping is not None:
        kw["mapping"] = eval(mapping)

    return kw


def args_coco_to_yolo(args=None):
    parser = ArgumentParser(description="yolo from coco")
    parser.add_argument("coco_dir", type=str,
                        help="dataset root dir")
    parser.add_argument("json_dir", type=str,
                        help="coco file dir")
    args = parser.parse_args(args=args)

    kw = vars(args)
    return kw


def args_coco_keep_p_sample(args=None):
    parser = ArgumentParser(description="Keep P Sample(s) In task")
    parser.add_argument("p", type=float,
                        help="p samples to keep")
    parser.add_argument("coco_file", type=str,
                        help="coco file path")
    parser.add_argument("--stratified", action="store_true",
                        help="stratified or not")
    parser.add_argument("--seed", type=int, default=1000,
                        help="random seed")
    args = parser.parse_args(args=args)

    kw = vars(args)
    return kw


def args_coco_leave_p_group(args=None):
    parser = ArgumentParser(description="Leave P Group(s) Out task")
    parser.add_argument("p", type=int,
                        help="p groups to leave")
    parser.add_argument("coco_file", type=str,
                        help="coco file path")
    args = parser.parse_args(args=args)

    kw = vars(args)
    return kw


def args_patch_images(args=None):
    parser = ArgumentParser(description="patch images")
    parser.add_argument("img_dir", type=str,
                        help="image dir")
    parser.add_argument("out_dir", type=str,
                        help="output dir")
    parser.add_argument("patch_size", type=int,
                        help="image patch size")
    parser.add_argument("-o", "--overlap", type=int, default=128,
                        help="the overlap area size")
    parser.add_argument("-m", "--color_mode", type=int, default=1,
                        help="color mode: 0=gray or 1=color")
    args = parser.parse_args(args=args)

    kw = vars(args)
    return kw


def args_display_coco(args=None):
    parser = ArgumentParser(description="display coco")
    parser.add_argument("coco_dir", type=str,
                        help="dataset root dir")
    parser.add_argument("coco_file", type=str,
                        help="coco file path")
    parser.add_argument("output_dir", type=str,
                        help="output dir")
    parser.add_argument("-o", "--options", type=str, default=None,
                        help="python dict, be run `eval(options)`")
    args = parser.parse_args(args=args)

    kw = vars(args)
    options = kw.pop("options")
    if options is not None:
        kw.update(eval(options))

    return kw


def args_display_test(args=None):
    parser = ArgumentParser(description="display test")
    parser.add_argument("results", type=str,
                        help="path of pkl file")
    parser.add_argument("score_thr", type=str,
                        help="python dict, be run `eval(score_thr)`")
    parser.add_argument("output_dir", type=str,
                        help="output dir")
    parser.add_argument("-o", "--options", type=str, default=None,
                        help="python dict, be run `eval(options)`")
    args = parser.parse_args(args=args)

    kw = vars(args)
    score_thr = kw.pop("score_thr")
    kw["score_thr"] = eval(score_thr)
    options = kw.pop("options")
    if options is not None:
        kw.update(eval(options))

    return kw


help_doc_str = """
Options:

positional arguments:
    command
        coco
        coco2yolo
        coco4kps
        coco4lpg
        patch
        viz-coco
        viz-test
"""


def _main(args=None):
    if args is None:
        args = sys.argv[1:]

    if len(args) >= 1:
        task, *args = args
    else:
        task, *args = ["--help"]

    if task == "coco":
        kw = args_coco_build(args)
        return coco_build(**kw)
    elif task == "coco2yolo":
        kw = args_coco_to_yolo(args)
        return coco_to_yolo(**kw)
    elif task == "coco4kps":
        kw = args_coco_keep_p_sample(args)
        p = kw.pop("p")
        return KeepPSamplesIn(p).split(**kw)
    elif task == "coco4lpg":
        kw = args_coco_leave_p_group(args)
        p = kw.pop("p")
        return LeavePGroupsOut(p).split(**kw)
    elif task == "patch":
        kw = args_patch_images(args)
        return patch_images(**kw)
    elif task == "viz-coco":
        kw = args_display_coco(args)
        return display_coco(**kw)
    elif task == "viz-test":
        kw = args_display_test(args)
        return display_test(**kw)
    elif task == "-h" or task == "--help":
        print("usage: python -m cvtk.utils.abc command ...\n", help_doc_str)
    else:
        print(f"unimplemented command: {task}\n", help_doc_str)


# develop:
# python cvtk/utils/abc ...
# runtime:
# python -m cvtk.utils.abc ...
if __name__ == "__main__":
    sys.exit(_main())
