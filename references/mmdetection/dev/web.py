import sys
import traceback

import tornado.ioloop
import tornado.web
from cvtk.io import imread
from cvtk.utils.abc.gen import image_label
from mmdet.datasets.pipelines import Compose

from py_app import inference_detector, init_detector

G_KW = dict(mode="max_score", score_thr={"*": 0.3}, label_grade={"*": 1})
G_MODEL = None
G_DEVICE = None
G_PIPLINE = None


class MainHandler(tornado.web.RequestHandler):

    def post(self):
        try:
            global G_KW, G_MODEL, G_DEVICE, G_PIPLINE

            file_list = self.get_arguments("image")
            imgs = [imread(uri, 1) for uri in file_list]
            results = inference_detector(imgs, G_MODEL, G_DEVICE, G_PIPLINE)
            labels = [image_label(dts, **G_KW) for dts in results]
            output = list(zip(file_list, labels, results))

            self.finish({"status": 0, "data": output})
        except Exception:
            output = traceback.format_exc()
            self.finish({"status": 1, "data": output})


class SettingsHandler(tornado.web.RequestHandler):

    def post(self):
        try:
            global G_KW

            mode = self.get_argument("mode", None)
            if mode is not None:
                G_KW["mode"] = mode

            for k in ["score_thr", "label_grade"]:
                v = self.get_argument(k, None)
                if v is not None:
                    G_KW[k] = eval(v)

            self.finish({"status": 0, "data": G_KW})
        except Exception:
            output = traceback.format_exc()
            self.finish({"status": 1, "data": output})


# CUDA_VISIBLE_DEVICES=1 python web.py 7000 config checkpoint
if __name__ == "__main__":
    args = sys.argv[1:]

    port = int(args[0])
    G_MODEL = init_detector(args[1], args[2], device="cuda:0")
    G_DEVICE = next(G_MODEL.parameters()).device

    cfg = G_MODEL.cfg.copy()
    G_PIPLINE = cfg.data.test.pipeline
    G_PIPLINE[0].type = "LoadImageFromWebcam"
    G_PIPLINE = Compose(G_PIPLINE)

    tornado.web.Application([
        (r"/predict", MainHandler),
        (r"/settings", SettingsHandler),
    ]).listen(port)
    tornado.ioloop.IOLoop.current().start()
