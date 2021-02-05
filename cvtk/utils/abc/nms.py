import copy
import torch
import numpy as np
import networkx as nx
from collections import defaultdict


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


def bbox_overlaps(dt, gt, mode="iou"):
    bboxes1 = torch.FloatTensor([d["xyxy"] for d in dt])
    bboxes2 = torch.FloatTensor([g["xyxy"] for g in gt])

    rows, cols = bboxes1.size(0), bboxes2.size(0)
    if rows * cols == 0:
        return None

    lt = torch.max(bboxes1[:, None, :2], bboxes2[:, :2])  # [rows, cols, 2]
    rb = torch.min(bboxes1[:, None, 2:], bboxes2[:, 2:])  # [rows, cols, 2]

    overlap = torch.prod((rb - lt).clamp(min=0), -1)

    area1 = (bboxes1[:, 2] - bboxes1[:, 0]) * (bboxes1[:, 3] - bboxes1[:, 1])
    area2 = (bboxes2[:, 2] - bboxes2[:, 0]) * (bboxes2[:, 3] - bboxes2[:, 1])

    if mode == "iou":
        ious = overlap / (area1[:, None] + area2 - overlap + 1.0)
    elif mode == "min":
        ious = overlap / (torch.min(area1[:, None], area2) + 1.0)
    elif mode == "max":
        ious = overlap / (torch.max(area1[:, None], area2) + 1.0)
    elif mode == "/dt":
        ious = overlap / (area1[:, None] + 1.0)
    elif mode == "/gt":
        ious = overlap / (area2[None, :] + 1.0)

    return ious.numpy()


def _collate_fn(nodes, lines, dt):
    dt_ = []
    for i_set in clustering(nodes, lines):
        vals = [dt[i] for i in i_set]
        best = max(vals, key=lambda x: x["score"])
        bboxes = np.array([d["xyxy"] for d in vals], dtype=np.float32)
        (x1, y1), (x2, y2) = bboxes[:, :2].min(axis=0).tolist(), bboxes[:, 2:].max(axis=0).tolist()
        best["bbox"] = [x1, y1, x2 - x1, y2 - y1]
        best["area"] = (x2 - x1) * (y2 - y1)
        best["xyxy"] = [x1, y1, x2, y2]
        dt_.append(best)
    return dt_


def _clean_with_ious(dt, mode="iou", thr=0.1):
    ious = bbox_overlaps(dt, dt, mode)

    nodes = list(range(ious.shape[0]))
    lines = np.argwhere(ious >= thr).tolist()

    return _collate_fn(nodes, lines, dt)


def _clean_with_dist(dt, k=1.5):
    bboxes = [d["xyxy"] for d in dt]

    bboxes = torch.FloatTensor(bboxes)
    xy = (bboxes[:, 2:] + bboxes[:, :2]) * 0.5
    wh = (bboxes[:, 2:] - bboxes[:, :2]).clamp(min=0)

    dist = (xy[:, None] - xy).abs()
    limit = (wh[:, None] + wh) * k + 1
    mask = torch.prod(dist <= limit, -1)

    nodes = list(range(mask.size(0)))
    lines = mask.nonzero(as_tuple=False).tolist()

    return _collate_fn(nodes, lines, dt)


def clean_by_bbox(dt, mode="dist", param=1.0):
    if len(dt) == 0:
        return dt

    dt = copy.deepcopy(dt)
    groups = defaultdict(list)
    for d in dt:
        w, h = d["bbox"][2:]
        if w * h >= 16:
            groups[d["label"]].append(d)

    dt_ = []
    for v in groups.values():
        if mode == "dist":
            dt_.extend(_clean_with_dist(v, param))
        else:
            dt_.extend(_clean_with_ious(v, mode, param))
    return dt_
