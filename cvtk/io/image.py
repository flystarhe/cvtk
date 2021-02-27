import re

import cv2 as cv
import numpy as np
import requests

URL_REGEX = re.compile(r"http://|https://|ftp://")


def imread(uri, flags=1):
    # flags(0: grayscale, 1: color)
    if isinstance(uri, str):
        if URL_REGEX.match(uri):
            buffer = requests.get(uri).content
            nparr = np.frombuffer(buffer, np.uint8)
            return cv.imdecode(nparr, flags)
        return cv.imread(uri, flags)

    if isinstance(uri, bytes):
        nparr = np.frombuffer(uri, np.uint8)
        return cv.imdecode(nparr, flags)

    raise Exception(f"{type(uri)} not supported")


def convert_color_factory(src, dst):
    code = getattr(cv, f"COLOR_{src.upper()}2{dst.upper()}")

    def convert_color(img):
        out_img = cv.cvtColor(img, code)
        return out_img

    return convert_color


bgr2rgb = convert_color_factory("bgr", "rgb")
rgb2bgr = convert_color_factory("rgb", "bgr")
