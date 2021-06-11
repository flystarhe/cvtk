import copy
import traceback

TMPL = dict(
    final=dict(
        img_cls=[],
        img_box=[],
        img_score=[],
        defect=1,
        status=200,
        message="",
        savepath="",
        show_path="",
        detect_cost_time="",
    )
)


def analyze(result, detect_time, info, img):
    _res = copy.deepcopy(TMPL)
    try:
        if len(result) == 0:
            out_final = dict(bbox=[0, 0, 1, 1], label="__OK", score=1.0)
        else:
            out_final = max(result, key=lambda x: x["score"])

        dat = _res["final"]
        dat["img_box"] = [out_final["bbox"]]
        dat["img_cls"] = [out_final["label"]]
        dat["img_score"] = [out_final["score"]]
        dat["detect_cost_time"] = f"{detect_time * 1000:.2f}ms"
        _res["final"] = dat
    except Exception:
        _res["message"] = traceback.format_exc()
        _res["status"] = 700
    return _res
