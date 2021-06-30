import os
import sys
from argparse import ArgumentParser

# Remove "" and current working directory from the first entry of sys.path
if sys.path[0] in ("", os.getcwd()):
    sys.path.pop(0)

from cvtk.tools.folder import split_folder


def args_split_folder(args=None):
    parser = ArgumentParser(description="count image size")
    parser.add_argument("indir", type=str,
                        help="input dir")
    parser.add_argument("-o", "--outdir", type=str, default=None,
                        help="output dir")
    parser.add_argument("-r", "--ref_file", type=str, default=None,
                        help="csv/xlsx file path")
    parser.add_argument("-a", "--col_path", type=str, default=None,
                        help="column name with file name")
    parser.add_argument("-b", "--col_label", type=str, default=None,
                        help="column name with image label")
    parser.add_argument("-s", "--suffixes", type=str, default=".jpg",
                        help="specify the file suffixes to include")
    parser.add_argument("-l", "--level", type=int, default=None,
                        help="keep directory levels(=2)")
    args = parser.parse_args(args=args)

    kw = vars(args)
    return kw


help_doc_str = """
Options:

positional arguments:
    command
        split-folder
"""


def _main(args=None):
    if args is None:
        args = sys.argv[1:]

    if len(args) >= 1:
        task, *args = args
    else:
        task, *args = ["--help"]

    if task == "split-folder":
        kw = args_split_folder(args)
        print(f"kwargs: {kw}")
        return split_folder(**kw)
    elif task == "-h" or task == "--help":
        print("usage: python -m cvtk.tools command ...")
    else:
        print(f"unimplemented command: {task}")

    return help_doc_str


# develop:
# python cvtk/tools ...
# runtime:
# python -m cvtk.tools ...
if __name__ == "__main__":
    print(_main())
    sys.exit(0)
