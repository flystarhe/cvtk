import sys

import tornado.ioloop
import tornado.web
from mmdet.apis import inference_detector, init_detector


class MainHandler(tornado.web.RequestHandler):

    def post(self):
        try:
            image = self.get_argument("image")
            output = inference_detector(my_model, image)
            results = {"status": 0, "data": str(output)}
        except Exception:
            results = {"status": 1}
        self.finish(results)


def load_model(config, checkpoint):
    return init_detector(config, checkpoint, device="cuda:0")


# CUDA_VISIBLE_DEVICES=1 python app_tornado.py 7000
if __name__ == "__main__":
    args = sys.argv[1:]

    port = int(args[0])
    my_model = load_model(args[1], args[2])

    tornado.web.Application([
        (r"/predict", MainHandler),
    ]).listen(port)
    tornado.ioloop.IOLoop.current().start()
