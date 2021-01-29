import cv2 as cv
import numpy as np
import re
import requests


URL_REGEX = re.compile(r"http://|https://|ftp://")


def imread(uri, flags=1):
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
