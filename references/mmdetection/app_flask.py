import gc
import sys

from flask import Flask, jsonify, request
from gevent.pywsgi import WSGIServer
from mmdet.apis import inference_detector, init_detector

app = Flask(__name__)

config = ""
checkpoint = ""
my_model = init_detector(config, checkpoint, device="cuda:0")


@app.route("/predict", methods=["POST"])
def predict():
    try:
        image = request.form["image"]
        output = inference_detector(my_model, image)
        results = {"status": 0, "data": str(output)}
        if request.form.get("gc", "n") == "y":
            gc.collect()
    except Exception:
        results = {"status": 1}
    return jsonify(results)


# CUDA_VISIBLE_DEVICES=1 python app_flask.py 7000
if __name__ == "__main__":
    port = sys.argv[1]
    http_server = WSGIServer(("", int(port)), app)
    http_server.serve_forever()
