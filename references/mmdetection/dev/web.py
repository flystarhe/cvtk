import sys

import tornado.ioloop
import tornado.web
from py_app import inference_detector, init_detector


class MainHandler(tornado.web.RequestHandler):

    def post(self):
        try:
            global model, device, test_pipeline
            images = self.get_arguments("image")
            output = inference_detector(images, model, device, test_pipeline)
            results = {"status": 0, "data": str(output)}
        except Exception:
            results = {"status": 1}
        self.finish(results)


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
