import sys
import traceback

from flask import Flask, jsonify, request
from gevent.pywsgi import WSGIServer

app = Flask(__name__)


@app.route("/predict", methods=["POST"])
def predict():
    try:
        imgs = request.form.getlist("image")
        output = f"\n\ndemo: {type(imgs)} {imgs}"
        return jsonify({"status": 0, "data": output})
    except Exception:
        output = traceback.format_exc()
        return jsonify({"status": 1, "data": output})


# CUDA_VISIBLE_DEVICES=1 python app_flask.py 7000
if __name__ == "__main__":
    args = sys.argv[1:]

    port = int(args[0])

    http_server = WSGIServer(("", port), app)
    http_server.serve_forever()
