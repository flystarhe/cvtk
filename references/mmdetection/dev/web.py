import copy
import json
import sys
import time
import traceback
from importlib import util
from pathlib import Path

import tornado.ioloop
import tornado.web
from cvtk.io import imread
from mmdet.datasets.pipelines import Compose

from py_app import inference_detector, init_detector

G_MODEL = None
G_DEVICE = None
G_PIPLINE = None
G_TMPL = dict(
    status=200,
    message="",
    result="-",
)


class MainHandler(tornado.web.RequestHandler):

    def post(self):
        _res = copy.deepcopy(G_TMPL)
        try:
            global G_MODEL, G_DEVICE, G_PIPLINE

            data = json.loads(self.get_argument("data"))
            img = imread(data["image"]["color"]["path"], 1)

            start_time = time.time()
            result = inference_detector([img], G_MODEL, G_DEVICE, G_PIPLINE)[0]
            detect_time = time.time() - start_time

            _res["result"] = analyze(result, detect_time, data["info"], img)
        except Exception:
            _res["message"] = traceback.format_exc()
            _res["status"] = 600
        self.finish(_res)


# CUDA_VISIBLE_DEVICES=GPU_ID python web.py PORT MODEL_PATH
if __name__ == "__main__":
    args = sys.argv[1:]

    PORT = int(args[0])
    MODEL_PATH = Path(args[1])

    config = MODEL_PATH / "config.py"
    checkpoint = MODEL_PATH / "latest.pth"
    assert config.is_file() and checkpoint.is_file()
    config, checkpoint = str(config), str(checkpoint)

    try:
        name, location = "_web_extension", str(MODEL_PATH / "web_extension.py")
        spec = util.spec_from_file_location(name, location)
        module = util.module_from_spec(spec)
        spec.loader.exec_module(module)
        analyze = module.analyze
    except Exception:
        raise Exception(traceback.format_exc())

    G_MODEL = init_detector(config, checkpoint, device="cuda:0")
    G_DEVICE = next(G_MODEL.parameters()).device

    cfg = G_MODEL.cfg.copy()
    G_PIPLINE = cfg.data.test.pipeline
    G_PIPLINE[0].type = "LoadImageFromWebcam"
    G_PIPLINE = Compose(G_PIPLINE)

    tornado.web.Application([
        (r"/predict", MainHandler),
    ]).listen(PORT)
    tornado.ioloop.IOLoop.current().start()
