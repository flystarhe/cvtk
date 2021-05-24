import sys
import traceback

import cv2 as cv
import numpy as np

npmode = None
npimage = None
bgr_image = None
ix, iy = -1, -1


help_doc_str = """
Press the left mouse button to start selecting the area.

- press `q` to exit
- press `r` to clear
- press `m` change mode

x1,y1,x2,y2
channel,min,mean,median,max,q05,q25,q75,q95
"""


def _mode(npmode):
    colors = ["BGR", "HSV", "YCrCb"]
    return colors[(colors.index(npmode) + 1) % len(colors)]


def _text(text):
    global bgr_image
    font = cv.FONT_HERSHEY_SIMPLEX
    cv.rectangle(bgr_image, (0, 0), (300, 40), (255, 255, 255), -1)
    cv.putText(bgr_image, text, (20, 35), font, 1, (0, 0, 255), 2, cv.LINE_AA)


def _info(mat, q=None):
    nparr = mat.ravel()

    if q is None:
        q = [0.05, 0.25, 0.75, 0.95]

    r1 = [func(nparr) for func in (np.min, np.mean, np.median, np.max)]
    r2 = np.quantile(nparr, q, interpolation="nearest").tolist()
    return ",".join([f"{x:4d}" for x in r1 + r2])


def _on_mouse(event, x, y, flags, param):
    global npmode, npimage, bgr_image, ix, iy

    if event == cv.EVENT_LBUTTONDOWN:
        ix, iy = x, y
    elif event == cv.EVENT_MOUSEMOVE:
        vals = [f"{v:3d}" for v in npimage[y, x]]
        _text(f"{npmode} // {vals}")
        if ix >= 0 and iy >= 0:
            cv.rectangle(bgr_image, (ix, iy), (x, y), (0, 255, 0), -1)
    elif event == cv.EVENT_LBUTTONUP:
        if ix >= 0 and iy >= 0:
            cv.rectangle(bgr_image, (ix, iy), (x, y), (0, 255, 0), -1)
            vals = [f"{v:3d}" for v in (ix, iy, x, y)]
            print(f"{npmode} // {vals}")
            for i in range(3):
                print(_info(npimage[iy:y, ix:x, i]))
        ix, iy = -1, -1


def color_range(image_path):
    global npmode, npimage, bgr_image

    _nparr = cv.imread(image_path, 1)

    npmode = "BGR"
    npimage = _nparr.copy()
    bgr_image = _nparr.copy()

    print(help_doc_str)
    cv.imshow("Image", bgr_image)
    cv.setMouseCallback("Image", _on_mouse)

    while True:
        cv.imshow("Image", bgr_image)

        key = cv.waitKey(1) & 0xFF

        if key == ord("q"):
            break
        elif key == ord("r"):
            npmode = "BGR"
            npimage = _nparr.copy()
            bgr_image = _nparr.copy()
        elif key == ord("m"):
            npmode = _mode(npmode)
            code = getattr(cv, f"COLOR_BGR2{npmode}")
            npimage = cv.cvtColor(_nparr, code)
            _text(f"{npmode} //")
        elif key == 27:
            break

    cv.destroyAllWindows()

    return f"{image_path}"


def _main(args=None):
    try:
        if args is None:
            args = sys.argv[1:]
        return color_range(args[0])
    except Exception:
        e = traceback.format_exc()
        return f"\n\n{e}\n\n{sys.argv}"


if __name__ == "__main__":
    print(_main())
    sys.exit(0)
