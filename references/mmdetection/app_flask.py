import gc
import sys

from flask import Flask, jsonify, request
from mmdet.apis import inference_detector, init_detector


app = Flask(__name__)

config = ""
checkpoint = ""
my_model = init_detector(config, checkpoint, device="cuda:0")


@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json(force=True)
        output = inference_detector(my_model, data["image"])
        results = {"status": 0, : "data": output}
        if data.get("gc", "none") != "none":
            gc.collect()
    except Exception:
        results = {"status": 1}
    return jsonify(results)


# CUDA_VISIBLE_DEVICES=1 python app_flask.py 7000
if __name__ == "__main__":
    port = sys.argv[1]
    app.run("0.0.0.0", port)
