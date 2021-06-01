import copy
from collections import defaultdict

import networkx as nx
import numpy as np
import torch


def xyxy2xywh(bbox):
    x1, y1, x2, y2 = map(int, bbox)
    return [x1, y1, x2 - x1, y2 - y1]


def xywh2xyxy(bbox):
    x1, y1, w, h = map(int, bbox)
    return [x1, y1, x1 + w, y1 + h]


def clustering(nodes, lines):
    # G.add_nodes_from([1, 2, 3, 4])
    # G.add_edges_from([(1, 2), (1, 3)])
    G = nx.Graph()
    G.add_nodes_from(nodes)
    G.add_edges_from(lines)
    return nx.connected_components(G)


def bbox_overlaps(dts, gts, mode="iou"):
    bboxes1 = torch.FloatTensor([dt["xyxy"] for dt in dts])
    bboxes2 = torch.FloatTensor([gt["xyxy"] for gt in gts])

    rows, cols = bboxes1.size(0), bboxes2.size(0)
    if rows * cols == 0:
        return None

    lt = torch.max(bboxes1[:, None, :2], bboxes2[:, :2])  # [rows, cols, 2]
    rb = torch.min(bboxes1[:, None, 2:], bboxes2[:, 2:])  # [rows, cols, 2]

    overlap = torch.prod((rb - lt).clamp(min=0), -1)

    area1 = (bboxes1[:, 2] - bboxes1[:, 0]) * (bboxes1[:, 3] - bboxes1[:, 1])
    area2 = (bboxes2[:, 2] - bboxes2[:, 0]) * (bboxes2[:, 3] - bboxes2[:, 1])

    if mode == "iou":
        ious = overlap / (area1[:, None] + area2 - overlap + 1e-5)
    elif mode == "min":
        ious = overlap / (torch.min(area1[:, None], area2) + 1e-5)
    elif mode == "max":
        ious = overlap / (torch.max(area1[:, None], area2) + 1e-5)
    elif mode == "/dts":
        ious = overlap / (area1[:, None] + 1e-5)
    elif mode == "/gts":
        ious = overlap / (area2[None, :] + 1e-5)

    return ious.numpy()


def _collate_fn(nodes, lines, dts):
    dts_ = []
    for i_set in clustering(nodes, lines):
        vals = [dts[i] for i in i_set]
        best = max(vals, key=lambda x: x["score"])
        bboxes = np.array([dt["xyxy"] for dt in vals], dtype=np.float32)
        x1, y1 = bboxes[:, :2].min(axis=0).tolist()
        x2, y2 = bboxes[:, 2:].max(axis=0).tolist()
        best["bbox"] = [x1, y1, x2 - x1, y2 - y1]
        best["area"] = (x2 - x1) * (y2 - y1)
        best["xyxy"] = [x1, y1, x2, y2]
        dts_.append(best)
    return dts_


def _clean_with_ious(dts, mode="iou", thr=0.1):
    ious = bbox_overlaps(dts, dts, mode)

    nodes = list(range(ious.shape[0]))
    lines = np.argwhere(ious >= thr).tolist()

    return _collate_fn(nodes, lines, dts)


def _clean_with_dist(dts, k=1.0):
    bboxes = [dt["xyxy"] for dt in dts]

    bboxes = torch.FloatTensor(bboxes)
    xy = (bboxes[:, 2:] + bboxes[:, :2]) * 0.5
    wh = (bboxes[:, 2:] - bboxes[:, :2]).clamp(min=0)

    dist = (xy[:, None] - xy).abs()
    limit = (wh[:, None] + wh) * k + 1
    mask = torch.prod(dist <= limit, -1)

    nodes = list(range(mask.size(0)))
    lines = mask.nonzero(as_tuple=False).tolist()

    return _collate_fn(nodes, lines, dts)


def _clean_with_one(dts):
    best = max(dts, key=lambda x: x["score"])
    bboxes = np.array([dt["xyxy"] for dt in dts], dtype=np.float32)
    x1, y1 = bboxes[:, :2].min(axis=0).tolist()
    x2, y2 = bboxes[:, 2:].max(axis=0).tolist()
    best["bbox"] = [x1, y1, x2 - x1, y2 - y1]
    best["area"] = (x2 - x1) * (y2 - y1)
    best["xyxy"] = [x1, y1, x2, y2]
    return [best]


def clean_by_bbox(dts, mode="dist", param=1.0):
    if len(dts) == 0:
        return dts

    dts = copy.deepcopy(dts)
    groups = defaultdict(list)
    for dt in dts:
        w, h = dt["bbox"][2:]
        if w * h >= 16:
            groups[dt["label"]].append(dt)

    dts_ = []
    for v in groups.values():
        if mode == "one":
            dts_.extend(_clean_with_one(v))
        elif mode == "dist":
            dts_.extend(_clean_with_dist(v, param))
        else:
            dts_.extend(_clean_with_ious(v, mode, param))
    return dts_
