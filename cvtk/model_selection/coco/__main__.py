import os
import sys

from cvtk.model_selection.coco import KeepPSamplesIn, LeavePGroupsOut

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
        return KeepPSamplesIn(args[1]).split(args[2])
    elif args[0] == "lp":
        return LeavePGroupsOut(args[1]).split(args[2])
    else:
        raise NotImplementedError(f"Not supported args: {args}")


# develop:
# python cvtk/model_selection/coco [kp|lp]
# runtime:
# python -m cvtk.model_selection.coco [kp|lp] <samples/groups> <coco file path>
if __name__ == "__main__":
    sys.exit(_main())
