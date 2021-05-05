import sys

from flask import Flask, jsonify, request
from gevent.pywsgi import WSGIServer
from mmdet.apis import inference_detector, init_detector

app = Flask(__name__)


@app.route("/predict", methods=["POST"])
def predict():
    try:
        image = request.form["image"]
        output = inference_detector(my_model, image)
        results = {"status": 0, "data": str(output)}
    except Exception:
        results = {"status": 1}
    return jsonify(results)


def load_model(config, checkpoint):
    return init_detector(config, checkpoint, device="cuda:0")


# CUDA_VISIBLE_DEVICES=1 python app_flask.py 7000
if __name__ == "__main__":
    args = sys.argv[1:]

    port = int(args[0])
    my_model = load_model(args[1], args[2])

    http_server = WSGIServer(("", port), app)
    http_server.serve_forever()
