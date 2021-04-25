import gc
import sys

import tornado.ioloop
import tornado.web
from mmdet.apis import inference_detector, init_detector


class MainHandler(tornado.web.RequestHandler):

    def post(self):
        try:
            image = self.get_argument("image")
            output = inference_detector(my_model, image)
            results = {"status": 0, "data": output}
            if self.get_argument("gc", "none") != "none":
                gc.collect()
        except Exception:
            results = {"status": 1}
        self.finish(results)


def make_app():
    return tornado.web.Application([
        (r"/predict", MainHandler),
    ])


config = ""
checkpoint = ""
my_model = init_detector(config, checkpoint, device="cuda:0")


# CUDA_VISIBLE_DEVICES=1 python app_tornado.py 7000
if __name__ == "__main__":
    port = sys.argv[1]
    make_app().listen(int(port))
    tornado.ioloop.IOLoop.current().start()
