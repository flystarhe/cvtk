import os
import sys
from argparse import ArgumentParser

# Remove "" and current working directory from the first entry of sys.path
if sys.path[0] in ("", os.getcwd()):
    sys.path.pop(0)

from cvtk.utils.abc.coco.build import make_dataset as coco_build
from cvtk.utils.abc.coco.image import count_image_size as image_size
from cvtk.utils.abc.coco.sampling import KeepPSamplesIn, LeavePGroupsOut
from cvtk.utils.abc.coco.yolo import yolo_from_coco as coco_to_yolo
from cvtk.utils.abc.gen import gen_test
from cvtk.utils.abc.patch import patch_images
from cvtk.utils.abc.visualize import display_coco, display_test


def args_coco_build(args=None):
    parser = ArgumentParser(description="build dataset")
    parser.add_argument("img_dir", type=str,
                        help="images dir")
    parser.add_argument("-a", "--ann_dir", type=str, default=None,
                        help="labels dir, search annotation from here")
    parser.add_argument("-o", "--out_dir", type=str, default=None,
                        help="output dir, default save to `{img_dir}_coco`")
    parser.add_argument("-i", "--include", type=str, default=None,
                        help="filter dataset with hiplot(*.csv), coco(*.json) or dir(path/)")
    parser.add_argument("-m", "--mapping", type=str, default=None,
                        help="python dict, be run `eval(mapping)`")
    parser.add_argument("-e", "--min_size", type=int, default=0,
                        help="padding mini bbox to min size")
    parser.add_argument("--all-imgs", action="store_true",
                        help="keep none-shapes image")
    args = parser.parse_args(args=args)

    kw = vars(args)
    mapping = kw.pop("mapping")
    if mapping is not None:
        kw["mapping"] = eval(mapping)
    return kw


def args_image_size(args=None):
    parser = ArgumentParser(description="count image size")
    parser.add_argument("img_dir", type=str,
                        help="images dir")
    args = parser.parse_args(args=args)

    kw = vars(args)
    return kw


def args_coco_keep_p_sample(args=None):
    parser = ArgumentParser(description="Keep P Sample(s) In task")
    parser.add_argument("p", type=float,
                        help="p samples to keep")
    parser.add_argument("coco_file", type=str,
                        help="coco file path")
    parser.add_argument("-n", "--num_groups", type=int, default=1,
                        help="specify num groups for dataset")
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


def args_coco_to_yolo(args=None):
    parser = ArgumentParser(description="yolo from coco")
    parser.add_argument("coco_dir", type=str,
                        help="dataset root dir")
    parser.add_argument("json_dir", type=str,
                        help="coco file dir")
    args = parser.parse_args(args=args)

    kw = vars(args)
    return kw


def args_gen_test(args=None):
    parser = ArgumentParser(description="gen test")
    parser.add_argument("results", type=str,
                        help="path of pkl file")
    parser.add_argument("mode", type=str,
                        help="complex, max_score or rank_mixed")
    parser.add_argument("score_thr", type=str,
                        help="python dict, be run `eval(score_thr)`")
    parser.add_argument("label_grade", type=str,
                        help="python dict, be run `eval(label_grade)`")
    parser.add_argument("-o", "--options", type=str, default=None,
                        help="python dict, be run `eval(options)`")
    args = parser.parse_args(args=args)

    kw = vars(args)
    score_thr = kw.pop("score_thr")
    kw["score_thr"] = eval(score_thr)
    label_grade = kw.pop("label_grade")
    kw["label_grade"] = eval(label_grade)
    options = kw.pop("options")
    if options is not None:
        kw.update(eval(options))
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
        img-size
        coco4kps
        coco4lpg
        coco2yolo
        gen-test
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
        print(f"kwargs: {kw}")
        return coco_build(**kw)
    elif task == "img-size":
        kw = args_image_size(args)
        print(f"kwargs: {kw}")
        return image_size(**kw)
    elif task == "coco4kps":
        kw = args_coco_keep_p_sample(args)
        print(f"kwargs: {kw}")
        p = kw.pop("p")
        return KeepPSamplesIn(p).split(**kw)
    elif task == "coco4lpg":
        kw = args_coco_leave_p_group(args)
        print(f"kwargs: {kw}")
        p = kw.pop("p")
        return LeavePGroupsOut(p).split(**kw)
    elif task == "coco2yolo":
        kw = args_coco_to_yolo(args)
        print(f"kwargs: {kw}")
        return coco_to_yolo(**kw)
    elif task == "gen-test":
        kw = args_gen_test(args)
        print(f"kwargs: {kw}")
        return gen_test(**kw)
    elif task == "patch":
        kw = args_patch_images(args)
        print(f"kwargs: {kw}")
        return patch_images(**kw)
    elif task == "viz-coco":
        kw = args_display_coco(args)
        print(f"kwargs: {kw}")
        return display_coco(**kw)
    elif task == "viz-test":
        kw = args_display_test(args)
        print(f"kwargs: {kw}")
        return display_test(**kw)
    elif task == "-h" or task == "--help":
        print("usage: python -m cvtk command ...")
    else:
        print(f"unimplemented command: {task}")

    return help_doc_str


# develop:
# python cvtk ...
# runtime:
# python -m cvtk ...
if __name__ == "__main__":
    print(_main())
    sys.exit(0)
