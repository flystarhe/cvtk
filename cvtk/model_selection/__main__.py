import os
import sys

# Remove "" and current working directory from the first entry of sys.path
while True:
    if sys.path[0] in ("", os.getcwd()):
        sys.path.pop(0)
    else:
        break


def _main(args=None):
    if args is None:
        args = sys.argv[1:]

    if args[0] == "kp":
        return "move `cvtk.model_selection` to `cvtk.model_selection.coco`"
    elif args[0] == "lp":
        return "move `cvtk.model_selection` to `cvtk.model_selection.coco`"
    else:
        raise NotImplementedError(f"Not supported args: {args}")

    return "none"


# develop:
# python cvtk/model_selection [kp|lp]
# runtime:
# python -m cvtk.model_selection [kp|lp] <samples/groups> <coco file path>
if __name__ == "__main__":
    sys.exit(_main())
