# ensemble_from_submissions.py
# 두 개 이상의 제출 csv(ddq_swinl_1024_12e.csv, ddq_swinl_10_full.csv 등)를
# 박스 단위로 앙상블(WBF or NMS)해서 새로운 제출용 csv를 만드는 스크립트

import argparse
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd


# --------------------------
# IoU / NMS / WBF
# --------------------------

def iou_xyxy(a, b):
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b

    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)

    iw = max(0.0, inter_x2 - inter_x1)
    ih = max(0.0, inter_y2 - inter_y1)
    inter = iw * ih

    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter + 1e-12

    return inter / union


def nms_per_class(boxes, scores, iou_thr: float):
    if not boxes:
        return [], []

    idxs = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
    keep = []

    while idxs:
        cur = idxs.pop(0)
        keep.append(cur)
        rest = []
        for i in idxs:
            if iou_xyxy(boxes[cur], boxes[i]) < iou_thr:
                rest.append(i)
        idxs = rest

    return [boxes[i] for i in keep], [scores[i] for i in keep]


def weighted_box(boxes, scores):
    sw = sum(scores) + 1e-12
    x1 = sum(b[0] * s for b, s in zip(boxes, scores)) / sw
    y1 = sum(b[1] * s for b, s in zip(boxes, scores)) / sw
    x2 = sum(b[2] * s for b, s in zip(boxes, scores)) / sw
    y2 = sum(b[3] * s for b, s in zip(boxes, scores)) / sw
    return [x1, y1, x2, y2]


def wbf_per_class(all_boxes, all_scores, iou_thr=0.55):
    flat = []
    for boxes, scores in zip(all_boxes, all_scores):
        for b, s in zip(boxes, scores):
            flat.append((b, s))

    flat.sort(key=lambda x: x[1], reverse=True)

    clusters = []
    for b, s in flat:
        matched = False
        for c in clusters:
            if iou_xyxy(c["box"], b) >= iou_thr:
                c["boxes"].append(b)
                c["scores"].append(s)
                c["box"] = weighted_box(c["boxes"], c["scores"])
                matched = True
                break
        if not matched:
            clusters.append({"boxes": [b], "scores": [s], "box": b})

    fused_boxes = []
    fused_scores = []
    for c in clusters:
        fb = weighted_box(c["boxes"], c["scores"])
        # 기존: fs = sum(c["scores"]) / max(1, len(c["scores"]))
        fs = max(c["scores"])            # ← 여기로 수정
        fused_boxes.append(fb)
        fused_scores.append(fs)

    order = sorted(range(len(fused_scores)), key=lambda i: fused_scores[i], reverse=True)
    fused_boxes = [fused_boxes[i] for i in order]
    fused_scores = [fused_scores[i] for i in order]

    return fused_boxes, fused_scores


# --------------------------
# PredictionString <-> list
# --------------------------

def parse_prediction_string(ps: str):
    """
    'cls score x y w h cls score x y w h ...'
    형태의 문자열을 파싱해서
    [(cls, score, x1, y1, x2, y2), ...] 리스트로 반환.
    """
    if not isinstance(ps, str) or ps.strip() == "":
        return []

    parts = ps.strip().split()
    if len(parts) % 6 != 0:
        # 이상한 행은 그냥 버리지 말고, 가능한 만큼만 읽는다
        n = len(parts) // 6
        parts = parts[: n * 6]

    out = []
    for i in range(0, len(parts), 6):
        cls_id = int(float(parts[i]))
        score = float(parts[i + 1])
        x = float(parts[i + 2])
        y = float(parts[i + 3])
        w = float(parts[i + 4])
        h = float(parts[i + 5])
        x1, y1 = x, y
        x2, y2 = x + w, y + h
        out.append((cls_id, score, x1, y1, x2, y2))

    return out


def make_prediction_string(items, score_thr: float):
    """
    items: [(cls, score, x1, y1, x2, y2), ...]
    -> 'cls score x y w h ...'
    """
    parts = []
    for cls_id, score, x1, y1, x2, y2 in items:
        if score < score_thr:
            continue
        w = max(0.0, x2 - x1)
        h = max(0.0, y2 - y1)
        parts.extend([
            str(int(cls_id)),
            f"{score:.6f}",
            f"{x1:.2f}",
            f"{y1:.2f}",
            f"{w:.2f}",
            f"{h:.2f}",
        ])
    return " ".join(parts)


# --------------------------
# 기타 유틸
# --------------------------

def infer_cols(df: pd.DataFrame):
    cols = list(df.columns)
    lower = [c.lower() for c in cols]

    id_col = None
    pred_col = None

    for c, lc in zip(cols, lower):
        if "image" in lc and "id" in lc:
            id_col = c
            break
    if id_col is None:
        for c, lc in zip(cols, lower):
            if lc in ("id", "image", "imageid"):
                id_col = c
                break
    if id_col is None:
        id_col = cols[0]

    for c, lc in zip(cols, lower):
        if "prediction" in lc or lc == "predictionstring":
            pred_col = c
            break
    if pred_col is None:
        if len(cols) >= 2:
            pred_col = cols[1]
        else:
            raise ValueError(f"Prediction 컬럼을 찾을 수 없습니다: {cols}")

    return id_col, pred_col


# --------------------------
# 메인 로직
# --------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csvs", nargs="+", required=True,
                        help="앙상블할 제출 csv 경로들 (2개 이상)")
    parser.add_argument("--weights", nargs="+", type=float,
                        help="각 csv에 대한 가중치 (생략 시 전부 1)")
    parser.add_argument("--iou-thr", type=float, default=0.55)
    parser.add_argument("--score-thr", type=float, default=0.001)
    parser.add_argument("--use-wbf", action="store_true",
                        help="설정 시 WBF, 아니면 단순 NMS")
    parser.add_argument("--out-csv", type=str, default="submission_ensemble.csv")
    args = parser.parse_args()

    csv_paths = [Path(p) for p in args.csvs]
    for p in csv_paths:
        if not p.exists():
            raise FileNotFoundError(p)

    if args.weights is None:
        weights = [1.0] * len(csv_paths)
    else:
        if len(args.weights) != len(csv_paths):
            raise ValueError("csv 개수와 weights 개수가 다릅니다.")
        weights = args.weights

    dfs = [pd.read_csv(p) for p in csv_paths]
    id_cols = []
    pred_cols = []
    for df in dfs:
        i, p = infer_cols(df)
        id_cols.append(i)
        pred_cols.append(p)

    # image_id 정렬/일치 여부 확인 (여기서는 0번째 csv 기준으로 맞춘다)
    base_ids = dfs[0][id_cols[0]].tolist()
    for k in range(1, len(dfs)):
        ids_k = dfs[k][id_cols[k]].tolist()
        if base_ids != ids_k:
            raise ValueError(f"{csv_paths[0]} 와 {csv_paths[k]} 의 image_id 순서가 다릅니다.")

    image_ids = base_ids
    print("총 이미지 수:", len(image_ids))

    out_pred_strings = []

    # 이미지별로 앙상블
    for idx, img_id in enumerate(image_ids):
        # model별 예측을 클래스 단위로 모을 구조
        per_model_by_class: List[Dict[int, List[Tuple[List[float], float]]]] = []

        for df, id_col, pred_col, w in zip(dfs, id_cols, pred_cols, weights):
            row = df.iloc[idx]
            ps = row[pred_col]
            items = parse_prediction_string(ps)

            by_class: Dict[int, List[Tuple[List[float], float]]] = {}
            for cls_id, score, x1, y1, x2, y2 in items:
                # 모델별 weight 적용 (score * weight)
                by_class.setdefault(cls_id, []).append(([x1, y1, x2, y2],
                                                        float(score) * w))
            per_model_by_class.append(by_class)

        # 클래스별로 NMS 또는 WBF
        merged_items = []

        all_cls_ids = set()
        for d in per_model_by_class:
            all_cls_ids.update(d.keys())

        for cls_id in sorted(all_cls_ids):
            model_boxes = []
            model_scores = []

            for d in per_model_by_class:
                items = d.get(cls_id, [])
                boxes = [b for b, s in items]
                scores = [s for b, s in items]
                model_boxes.append(boxes)
                model_scores.append(scores)

            if args.use_wbf and len(csv_paths) > 1:
                boxes_fused, scores_fused = wbf_per_class(
                    model_boxes, model_scores, args.iou_thr
                )
            else:
                flat_boxes = []
                flat_scores = []
                for boxes, scores in zip(model_boxes, model_scores):
                    flat_boxes.extend(boxes)
                    flat_scores.extend(scores)
                boxes_fused, scores_fused = nms_per_class(
                    flat_boxes, flat_scores, args.iou_thr
                )

            for b, s in zip(boxes_fused, scores_fused):
                if s >= args.score_thr:
                    merged_items.append((cls_id, s, b[0], b[1], b[2], b[3]))
        merged_items.sort(key=lambda t: t[1], reverse=True)
        ps_out = make_prediction_string(merged_items, args.score_thr)
        out_pred_strings.append(ps_out)

    # 0번째 csv의 스키마를 그대로 사용해서 저장
    out_df = dfs[0].copy()
    _, base_pred_col = infer_cols(dfs[0])
    out_df[base_pred_col] = out_pred_strings

    out_path = Path(args.out_csv)
    out_df.to_csv(out_path, index=False)
    print("저장 완료:", out_path)


if __name__ == "__main__":
    main()
