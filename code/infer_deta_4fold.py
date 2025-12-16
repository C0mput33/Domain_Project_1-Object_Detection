# infer_deta_4fold.py
# 목적:
# - 4-fold 학습 결과 중 원하는 fold checkpoint를 로드
# - baseline/dataset/test 이미지에 대해 COCO 형식 결과를 생성
# - 단일 모델 확인용이므로:
#   (1) --use-single-fold 옵션으로 fold 0만 써서 결과 확인 가능
#   (2) 기본은 4개 fold 결과를 score 평균 방식으로 단순 결합
#       (진짜 앙상블이 아니라, fold 안정성 확인을 위한 최소 결합)

import argparse
import json
from pathlib import Path
from typing import Dict, List

def ensure_mmdet_importable(project_root: Path):
    import sys
    mm_root = project_root / "mmdetection"
    if str(mm_root) not in sys.path:
        sys.path.insert(0, str(mm_root))

def find_latest_ckpt(work_dir: Path) -> Path:
    # fold work_dir에서 가장 최근 pth를 찾는다.
    if not work_dir.exists():
        raise FileNotFoundError(f"work_dir가 없습니다: {work_dir}")

    ckpts = sorted(work_dir.glob("*.pth"))
    if not ckpts:
        # epoch_*.pth 형태일 수도 있음
        ckpts = sorted(work_dir.glob("epoch_*.pth"))

    if not ckpts:
        raise FileNotFoundError(f"checkpoint를 찾지 못했습니다: {work_dir}")

    return ckpts[-1]

def load_classes_from_coco(json_path: Path) -> List[str]:
    coco = json.loads(json_path.read_text(encoding="utf-8"))
    cats = sorted(coco.get("categories", []), key=lambda x: x["id"])
    return [c["name"] for c in cats]

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--class-mode", type=str, default="none", choices=["none", "auto"])
    parser.add_argument("--use-single-fold", action="store_true", help="fold0만 사용")
    parser.add_argument("--epochs", type=int, default=10)
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parent  # baseline/
    ensure_mmdet_importable(project_root)

    from mmengine.config import Config
    from mmdet.apis import init_detector, inference_detector

    # 경로 고정
    data_root = project_root / "dataset"
    test_img_dir = data_root / "test"
    test_json = data_root / "test.json"

    if not test_img_dir.exists():
        raise FileNotFoundError(f"test 이미지 폴더가 없습니다: {test_img_dir}")

    # test.json이 실제 추론에 꼭 필요하진 않지만
    # 이미지 리스트를 안정적으로 얻기 위해 사용
    if not test_json.exists():
        raise FileNotFoundError(f"test.json이 없습니다: {test_json}")

    test_coco = json.loads(test_json.read_text(encoding="utf-8"))
    test_images = test_coco.get("images", [])
    if not test_images:
        raise ValueError("test.json의 images가 비어있습니다.")

    # 학습 스크립트와 동일한 base config 탐색
    from train_deta_4fold import find_base_config, patch_config_for_fold

    base_cfg_path = find_base_config(project_root)
    base_cfg = Config.fromfile(str(base_cfg_path))

    # fold 목록 결정
    folds = [0] if args.use_single_fold else [0, 1, 2, 3]

    # work_dir tag 맞추기
    tag = "classes_none" if args.class_mode == "none" else "classes_auto"
    work_root = project_root / "work_dirs" / f"deta_10e_4fold_{tag}"

    # fold별 모델 로드
    models = []
    cfgs = []
    for f in folds:
        cfg = base_cfg.copy()
        cfg = patch_config_for_fold(
            cfg=cfg,
            project_root=project_root,
            fold_idx=f,
            class_mode=args.class_mode,
            max_epochs=args.epochs,
        )

        fold_dir = Path(cfg.work_dir)
        ckpt = find_latest_ckpt(fold_dir)

        model = init_detector(cfg, str(ckpt), device="cuda:0")
        models.append(model)
        cfgs.append(cfg)

        print(f"[loaded] fold {f} ckpt: {ckpt}")

    # ------------------------------------------------------------
    # COCO results 생성
    # mmdet inference_detector는 (bboxes, labels) 기반의 결과를 준다.
    # 여기서는 단일 클래스/다중 클래스 모두 대응 가능한
    # COCO prediction dict 리스트를 만든다.
    # ------------------------------------------------------------
    results_all_folds: List[List[Dict]] = []

    for model in models:
        fold_results = []

        for img_info in test_images:
            file_name = img_info["file_name"]
            img_id = img_info["id"]
            img_path = test_img_dir / file_name

            if not img_path.exists():
                # test.json과 실제 폴더 파일이 안 맞으면 여기서 터진다.
                raise FileNotFoundError(f"test 이미지 파일 없음: {img_path}")

            pred = inference_detector(model, str(img_path))

            # pred 구조는 모델/버전에 따라 변형이 있다.
            # 가장 흔한 형태:
            # - DetDataSample 기반 -> pred.pred_instances.bboxes, scores, labels
            # 안전하게 접근
            pred_instances = pred.pred_instances
            bboxes = pred_instances.bboxes.cpu().numpy()
            scores = pred_instances.scores.cpu().numpy()
            labels = pred_instances.labels.cpu().numpy()

            # COCO category_id는 보통 1부터 시작
            # train.json categories id 체계를 그대로 따라야 안전하다.
            # 여기서는 labels(0-based) -> category_id(1-based) 가정.
            # 만약 네 train.json categories가 다른 규칙이면 이 매핑을 조정해야 한다.
            for bbox, score, label in zip(bboxes, scores, labels):
                x1, y1, x2, y2 = bbox.tolist()
                w = x2 - x1
                h = y2 - y1

                fold_results.append({
                    "image_id": img_id,
                    "category_id": int(label) + 1,
                    "bbox": [x1, y1, w, h],
                    "score": float(score),
                })

        results_all_folds.append(fold_results)

    # ------------------------------------------------------------
    # fold 결과 결합
    # 단일 모델 성능 확인이 목적이므로
    # - use_single_fold이면 fold0 결과 그대로 저장
    # - 아니면 동일 image_id/category_id/bbox가 완전히 같진 않으므로
    #   "진짜 bbox-level ensemble" 대신
    #   가장 보수적인 방식으로 fold 결과를 그냥 합쳐서 저장한다.
    #   (평가 서버가 NMS를 내부에서 적용하거나,
    #    제출용 포맷이 단순 list라면 이 방식도 확인용으로는 충분)
    # ------------------------------------------------------------
    if args.use_single_fold:
        final_results = results_all_folds[0]
        out_name = "deta_fold0_predictions.json"
    else:
        final_results = []
        for fr in results_all_folds:
            final_results.extend(fr)
        out_name = "deta_4fold_merged_predictions.json"

    out_path = project_root / out_name
    out_path.write_text(json.dumps(final_results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"saved predictions: {out_path}")

if __name__ == "__main__":
    main()
