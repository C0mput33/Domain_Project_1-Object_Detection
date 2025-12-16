# infer_deta_4fold_ensemble_csv.py
# 목적:
# - DETA 4-fold 학습 결과(best checkpoint)를 자동 수집
# - test 이미지에 대해 fold별 예측 수행
# - 이미지 단위로 NMS 또는 WBF로 통합
# - 결과를 범용 CSV로 저장
#
# 중요:
# - class-mode/epochs는 학습 때 사용한 값과 동일해야 한다.
# - 대회 submission 스키마가 따로 있다면 CSV 저장부만 교체하면 된다.

import argparse
import csv
import json
from pathlib import Path
from typing import List, Dict, Tuple


def find_base_config(project_root: Path) -> Path:
    """
    학습 코드와 동일한 방식으로 base config를 찾는다.
    """
    mm_root = project_root / "mmdetection"
    cfg_root = mm_root / "configs"

    BASE_CFG_CANDIDATES = [
        cfg_root / "deta" / "deta_swin-large_8xb2-12e_coco.py",
        cfg_root / "deta" / "deta_swin-large_8xb2-24e_coco.py",
        cfg_root / "deta" / "deta_swin-large_8xb2-36e_coco.py",
    ]

    for c in BASE_CFG_CANDIDATES:
        if c.exists():
            return c

    if cfg_root.exists():
        all_cfgs = list(cfg_root.rglob("*.py"))
        scored = []
        for p in all_cfgs:
            name = p.name.lower()
            score = 0
            if "deta" in name:
                score += 5
            if "swin" in name:
                score += 3
            if "large" in name or "l" in name:
                score += 2
            if "coco" in name:
                score += 1
            if score > 0:
                scored.append((score, p))

        scored.sort(key=lambda x: (-x[0], str(x[1])))
        if scored:
            return scored[0][1]

    raise FileNotFoundError("DETA base config를 찾지 못했습니다.")


def ensure_mmdet_importable(project_root: Path):
    """
    baseline/mmdetection import 경로를 활성화한다.
    """
    mm_root = project_root / "mmdetection"
    if not mm_root.exists():
        raise FileNotFoundError(f"mmdetection 폴더가 없습니다: {mm_root}")

    import sys
    if str(mm_root) not in sys.path:
        sys.path.insert(0, str(mm_root))


def load_classes_from_coco(json_path: Path) -> List[str]:
    """
    COCO categories에서 classes를 id 오름차순으로 추출한다.
    """
    with json_path.open("r", encoding="utf-8") as f:
        coco = json.load(f)

    cats = coco.get("categories", [])
    if not cats:
        raise ValueError("categories가 비어있습니다.")
    cats = sorted(cats, key=lambda x: x["id"])
    return [c["name"] for c in cats]


def _remove_metainfo_if_exists(dataset_cfg):
    """
    metainfo 제거 헬퍼.
    """
    try:
        dataset_cfg.pop("metainfo", None)
    except Exception:
        pass


def _set_metainfo_classes(dataset_cfg, classes: List[str]):
    """
    metainfo.classes 주입 헬퍼.
    """
    dataset_cfg.metainfo = dict(classes=tuple(classes))


def infer_img_prefix_from_coco(ann_path: Path) -> str:
    """
    file_name이 순수 파일명인지, 'train/xxx.jpg' 형태인지에 따라
    data_prefix 중복을 피하기 위한 간단 판별기.
    """
    with ann_path.open("r", encoding="utf-8") as f:
        coco = json.load(f)

    images = coco.get("images", [])
    if not images:
        return "train/"

    fn = images[0].get("file_name", "")
    if "/" in fn or "\\" in fn:
        return ""
    return "train/"


def patch_config_for_fold_infer(cfg,
                                project_root: Path,
                                fold_idx: int,
                                class_mode: str,
                                max_epochs: int):
    """
    추론 시에도 학습과 동일한 class/metainfo/num_classes 규칙을 유지하기 위한 패치.
    """
    data_root = Path("/data/ephemeral/home/model/dataset")
    folds_dir = data_root / "folds"

    train_ann = folds_dir / f"train_fold{fold_idx}.json"
    val_ann = folds_dir / f"val_fold{fold_idx}.json"

    if not train_ann.exists() or not val_ann.exists():
        raise FileNotFoundError("fold annotation json이 없습니다.")

    cfg.data_root = str(data_root)

    train_prefix = infer_img_prefix_from_coco(train_ann)
    val_prefix = infer_img_prefix_from_coco(val_ann)

    if "train_dataloader" in cfg:
        cfg.train_dataloader.dataset.data_root = str(data_root)
        cfg.train_dataloader.dataset.ann_file = str(train_ann)
        cfg.train_dataloader.dataset.data_prefix = dict(img=train_prefix)

        if class_mode == "none":
            _remove_metainfo_if_exists(cfg.train_dataloader.dataset)
        else:
            classes = load_classes_from_coco(train_ann)
            _set_metainfo_classes(cfg.train_dataloader.dataset, classes)

    if "val_dataloader" in cfg:
        cfg.val_dataloader.dataset.data_root = str(data_root)
        cfg.val_dataloader.dataset.ann_file = str(val_ann)
        cfg.val_dataloader.dataset.data_prefix = dict(img=val_prefix)

        if class_mode == "none":
            _remove_metainfo_if_exists(cfg.val_dataloader.dataset)
        else:
            classes = load_classes_from_coco(val_ann)
            _set_metainfo_classes(cfg.val_dataloader.dataset, classes)

    # num_classes 동기화
    classes = None
    try:
        classes = load_classes_from_coco(train_ann)
    except Exception:
        classes = None

    if classes is not None and "model" in cfg:
        num_classes = len(classes)
        head = cfg.model.bbox_head

        if isinstance(head, dict) and "num_classes" in head:
            head["num_classes"] = num_classes

        if isinstance(head, list):
            for h in head:
                if isinstance(h, dict) and "num_classes" in h:
                    h["num_classes"] = num_classes

    cfg.train_cfg = dict(
        type="EpochBasedTrainLoop",
        max_epochs=max_epochs,
        val_interval=1
    )

    work_root = project_root / "work_dirs"
    tag = "classes_none" if class_mode == "none" else "classes_auto"
    cfg.work_dir = str(work_root / f"deta_10e_4fold_{tag}" / f"fold{fold_idx}")

    return cfg


def find_best_checkpoint_for_fold(work_dir: Path) -> Path:
    """
    work_dir에서 best checkpoint를 최대한 견고하게 찾는다.
    """
    if not work_dir.exists():
        raise FileNotFoundError(f"work_dir 없음: {work_dir}")

    candidates = []
    candidates += list(work_dir.glob("best_*.pth"))
    candidates += list(work_dir.glob("*best*.pth"))

    if not candidates:
        epochs = sorted(work_dir.glob("epoch_*.pth"))
        if epochs:
            return epochs[-1]

    if not candidates:
        raise FileNotFoundError(f"best checkpoint를 찾지 못함: {work_dir}")

    candidates = sorted(candidates, key=lambda p: (len(p.name), p.name))
    return candidates[0]


def collect_test_images(data_root: Path) -> List[Path]:
    """
    test 이미지 폴더 자동 탐색 후 이미지 리스트를 반환한다.
    """
    test_dir_candidates = [
        data_root / "test",
        data_root / "Test",
        data_root / "images" / "test",
    ]

    test_dir = None
    for c in test_dir_candidates:
        if c.exists():
            test_dir = c
            break

    if test_dir is None:
        raise FileNotFoundError("test 이미지 폴더를 찾지 못했습니다.")

    exts = {".jpg", ".jpeg", ".png", ".bmp"}
    imgs = [p for p in sorted(test_dir.rglob("*")) if p.suffix.lower() in exts]

    if not imgs:
        raise FileNotFoundError("test 이미지가 비어있습니다.")

    return imgs


def parse_mmdet_result(result, score_thr: float) -> Dict[int, Dict[str, List]]:
    """
    inference_detector 출력(DetDataSample)을
    {class_id: {boxes:[], scores:[]}} 구조로 변환한다.
    """
    out = {}

    if hasattr(result, "pred_instances"):
        inst = result.pred_instances
        bboxes = inst.bboxes.cpu().numpy().tolist()
        scores = inst.scores.cpu().numpy().tolist()
        labels = inst.labels.cpu().numpy().tolist()

        for b, s, l in zip(bboxes, scores, labels):
            if s < score_thr:
                continue
            if l not in out:
                out[l] = {"boxes": [], "scores": []}
            out[l]["boxes"].append(b)
            out[l]["scores"].append(s)

    return out


def iou_xyxy(a: List[float], b: List[float]) -> float:
    """
    IoU 계산.
    """
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b

    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)

    inter_w = max(0.0, inter_x2 - inter_x1)
    inter_h = max(0.0, inter_y2 - inter_y1)
    inter = inter_w * inter_h

    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter

    if union <= 0:
        return 0.0
    return inter / union


def nms_per_class(boxes: List[List[float]],
                  scores: List[float],
                  iou_thr: float) -> List[int]:
    """
    단일 클래스 NMS.
    """
    idxs = sorted(range(len(boxes)), key=lambda i: scores[i], reverse=True)
    keep = []

    while idxs:
        i = idxs.pop(0)
        keep.append(i)

        new_idxs = []
        for j in idxs:
            if iou_xyxy(boxes[i], boxes[j]) < iou_thr:
                new_idxs.append(j)
        idxs = new_idxs

    return keep


def simple_wbf_per_class(boxes_list: List[List[List[float]]],
                         scores_list: List[List[float]],
                         weights: List[float],
                         iou_thr: float) -> Tuple[List[List[float]], List[float]]:
    """
    간단 WBF 구현.
    """
    merged = []
    for m_idx, (bxs, scs) in enumerate(zip(boxes_list, scores_list)):
        w = weights[m_idx]
        for b, s in zip(bxs, scs):
            merged.append((b, s * w, w))

    merged.sort(key=lambda x: x[1], reverse=True)

    fused_boxes = []
    fused_scores = []
    used = [False] * len(merged)

    for i in range(len(merged)):
        if used[i]:
            continue

        base_box, base_score, base_w = merged[i]
        cluster_boxes = [base_box]
        cluster_scores = [base_score]
        used[i] = True

        for j in range(i + 1, len(merged)):
            if used[j]:
                continue
            b, s, w = merged[j]
            if iou_xyxy(base_box, b) >= iou_thr:
                used[j] = True
                cluster_boxes.append(b)
                cluster_scores.append(s)

        coord_weights = cluster_scores
        sum_w = sum(coord_weights) if coord_weights else 1.0

        x1 = sum(b[0] * cw for b, cw in zip(cluster_boxes, coord_weights)) / sum_w
        y1 = sum(b[1] * cw for b, cw in zip(cluster_boxes, coord_weights)) / sum_w
        x2 = sum(b[2] * cw for b, cw in zip(cluster_boxes, coord_weights)) / sum_w
        y2 = sum(b[3] * cw for b, cw in zip(cluster_boxes, coord_weights)) / sum_w

        fused_boxes.append([x1, y1, x2, y2])
        fused_scores.append(sum(cluster_scores) / max(1, len(cluster_scores)))

    return fused_boxes, fused_scores


def ensemble_predictions(
    preds_per_model: List[Dict[int, Dict[str, List]]],
    num_classes: int,
    ensemble_type: str,
    iou_thr: float,
    weights: List[float]
) -> Dict[int, Dict[str, List]]:
    """
    이미지 1장에 대해 fold별 예측을 클래스 단위로 통합한다.
    """
    out = {}

    for c in range(num_classes):
        boxes_list = []
        scores_list = []

        for m in preds_per_model:
            pack = m.get(c, {"boxes": [], "scores": []})
            boxes_list.append(pack["boxes"])
            scores_list.append(pack["scores"])

        if all(len(b) == 0 for b in boxes_list):
            continue

        if ensemble_type == "nms":
            flat_boxes = []
            flat_scores = []
            for bxs, scs in zip(boxes_list, scores_list):
                flat_boxes.extend(bxs)
                flat_scores.extend(scs)

            keep = nms_per_class(flat_boxes, flat_scores, iou_thr)
            out[c] = {
                "boxes": [flat_boxes[i] for i in keep],
                "scores": [flat_scores[i] for i in keep],
            }

        elif ensemble_type == "wbf":
            fused_boxes, fused_scores = simple_wbf_per_class(
                boxes_list=boxes_list,
                scores_list=scores_list,
                weights=weights,
                iou_thr=iou_thr
            )
            out[c] = {"boxes": fused_boxes, "scores": fused_scores}

        else:
            raise ValueError("ensemble_type은 nms 또는 wbf만 허용")

    return out


def write_generic_csv(rows: List[Dict], out_csv: Path):
    """
    범용 CSV 저장.
    컬럼:
    file_name, class_id, score, x1, y1, x2, y2

    실제 대회 제출 스키마가 다르면 이 함수만 교체하면 된다.
    """
    fieldnames = ["file_name", "class_id", "score", "x1", "y1", "x2", "y2"]

    with out_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in rows:
            writer.writerow(r)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--class-mode", type=str, default="none", choices=["none", "auto"])
    parser.add_argument("--epochs", type=int, default=10)

    parser.add_argument("--ensemble", type=str, default="wbf", choices=["nms", "wbf"])
    parser.add_argument("--iou-thr", type=float, default=0.55)
    parser.add_argument("--score-thr", type=float, default=0.001)

    parser.add_argument("--out-csv", type=str, default="deta_4fold_ensemble.csv")

    args = parser.parse_args()

    project_root = Path("/data/ephemeral/home/model/baseline")
    data_root = Path("/data/ephemeral/home/model/dataset")

    ensure_mmdet_importable(project_root)

    from mmengine.config import Config
    from mmdet.apis import init_detector, inference_detector

    base_cfg_path = find_base_config(project_root)

    # fold별 config/ckpt 준비
    fold_cfgs = []
    fold_ckpts = []

    tag = "classes_none" if args.class_mode == "none" else "classes_auto"
    work_root = project_root / "work_dirs"
    fold_root = work_root / f"deta_10e_4fold_{tag}"

    for fold_idx in range(4):
        cfg = Config.fromfile(str(base_cfg_path))

        cfg = patch_config_for_fold_infer(
            cfg=cfg,
            project_root=project_root,
            fold_idx=fold_idx,
            class_mode=args.class_mode,
            max_epochs=args.epochs
        )

        fold_cfgs.append(cfg)

        work_dir = Path(cfg.work_dir)
        best_ckpt = find_best_checkpoint_for_fold(work_dir)
        fold_ckpts.append(best_ckpt)

    # num_classes 결정
    num_classes = None
    if args.class_mode == "auto":
        train_ann0 = data_root / "folds" / "train_fold0.json"
        classes = load_classes_from_coco(train_ann0)
        num_classes = len(classes)

    if num_classes is None:
        try:
            head = fold_cfgs[0].model.bbox_head
            if isinstance(head, dict) and "num_classes" in head:
                num_classes = int(head["num_classes"])
        except Exception:
            pass

    if num_classes is None:
        raise RuntimeError("num_classes를 결정하지 못했습니다. class-mode auto를 권장합니다.")

    # fold별 모델 초기화
    models = []
    for cfg, ckpt in zip(fold_cfgs, fold_ckpts):
        model = init_detector(cfg, str(ckpt), device="cuda:0")
        models.append(model)

    # test 이미지 수집
    test_imgs = collect_test_images(data_root)

    # fold 가중치(동일 가중)
    weights = [1.0, 1.0, 1.0, 1.0]

    # 최종 CSV row 모음
    final_rows = []

    for img_path in test_imgs:
        preds_per_model = []

        for model in models:
            result = inference_detector(model, str(img_path))
            pred_dict = parse_mmdet_result(result, score_thr=args.score_thr)
            preds_per_model.append(pred_dict)

        ens = ensemble_predictions(
            preds_per_model=preds_per_model,
            num_classes=num_classes,
            ensemble_type=args.ensemble,
            iou_thr=args.iou_thr,
            weights=weights
        )

        # 범용 CSV row 생성
        for cls_id, pack in ens.items():
            for box, score in zip(pack["boxes"], pack["scores"]):
                x1, y1, x2, y2 = box

                final_rows.append({
                    "file_name": img_path.name,
                    "class_id": int(cls_id),
                    "score": float(score),
                    "x1": float(x1),
                    "y1": float(y1),
                    "x2": float(x2),
                    "y2": float(y2),
                })

    out_csv = Path(args.out_csv)
    write_generic_csv(final_rows, out_csv)

    print(f"[OK] saved csv: {out_csv}")
    print("[INFO] used ckpts:")
    for i, p in enumerate(fold_ckpts):
        print(f"  fold{i}: {p}")


if __name__ == "__main__":
    main()
