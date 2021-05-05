import sys
import traceback

import tornado.ioloop
import tornado.web
from cvtk.io import imread
from mmdet.datasets.pipelines import Compose

from py_app import inference_detector, init_detector


class MainHandler(tornado.web.RequestHandler):

    def post(self):
        try:
            global model, device, test_pipeline
            imgs = [imread(f, 1) for f in self.get_arguments("image")]
            output = inference_detector(imgs, model, device, test_pipeline)
            self.finish({"status": 0, "data": output})
        except Exception:
            output = traceback.format_exc()
            self.finish({"status": 1, "data": output})


# CUDA_VISIBLE_DEVICES=1 python app_tornado.py 7000
if __name__ == "__main__":
    args = sys.argv[1:]

    port = int(args[0])
    model = init_detector(args[1], args[2], device="cuda:0")
    device = next(model.parameters()).device

    cfg = model.cfg.copy()
    test_pipeline = cfg.data.test.pipeline
    test_pipeline[0].type = "LoadImageFromWebcam"
    test_pipeline = Compose(test_pipeline)

    tornado.web.Application([
        (r"/predict", MainHandler),
    ]).listen(port)
    tornado.ioloop.IOLoop.current().start()
