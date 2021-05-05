import sys
import traceback

import tornado.ioloop
import tornado.web


class MainHandler(tornado.web.RequestHandler):

    def post(self):
        try:
            imgs = self.get_arguments("image")
            output = f"\n\ndemo: {type(imgs)} {imgs}"
            self.finish({"status": 0, "data": output})
        except Exception:
            output = traceback.format_exc()
            self.finish({"status": 1, "data": output})


# CUDA_VISIBLE_DEVICES=1 python app_tornado.py 7000
if __name__ == "__main__":
    args = sys.argv[1:]

    port = int(args[0])

    tornado.web.Application([
        (r"/predict", MainHandler),
    ]).listen(port)
    tornado.ioloop.IOLoop.current().start()
