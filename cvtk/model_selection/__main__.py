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

    return str(args)


# develop:
# python cvtk/model_selection ...
# runtime:
# python -m cvtk.model_selection ...
if __name__ == "__main__":
    sys.exit(_main())
