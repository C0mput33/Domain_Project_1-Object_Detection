# train_deta_4fold_v3.py
# 목적:
# - DETA 단일 모델 4-fold 학습
# - 10e 스케줄
# - class-mode auto/none 지원
# - 경로 중복(train/train) 자동 방지
# - AMP는 기본 OFF(DDQ/DETA 계열 half/float NMS 이슈 예방)
# - grad accumulation 지원

import argparse
import json
from pathlib import Path
from typing import List, Optional
import copy


def find_base_config(project_root: Path) -> Path:
    mm_root = project_root / "mmdetection"
    cfg_root = mm_root / "configs"

    BASE_CFG_CANDIDATES = [
        cfg_root / "deta" / "deta_swin-large_8xb2-12e_coco.py",
        cfg_root / "deta" / "deta_swin-large_8xb2-24e_coco.py",
        cfg_root / "deta" / "deta_swin-large_8xb2-36e_coco.py",
        # 메모리 압박이 계속되면 아래 후보를 주석 해제해서 우선 탐색하도록 변경 가능
        # cfg_root / "deta" / "deta_swin-base_8xb2-12e_coco.py",
        # cfg_root / "deta" / "deta_r50_8xb2-12e_coco.py",
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

    raise FileNotFoundError("DETA base config 탐색 실패")


def load_coco(json_path: Path) -> dict:
    with json_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def load_classes_from_coco(json_path: Path) -> List[str]:
    coco = load_coco(json_path)
    cats = coco.get("categories", [])
    if not cats:
        raise ValueError("categories가 비어있습니다.")
    cats = sorted(cats, key=lambda x: x["id"])
    return [c["name"] for c in cats]


def ensure_mmdet_importable(project_root: Path):
    mm_root = project_root / "mmdetection"
    if not mm_root.exists():
        raise FileNotFoundError(f"mmdetection 폴더가 없습니다: {mm_root}")

    import sys
    if str(mm_root) not in sys.path:
        sys.path.insert(0, str(mm_root))


def _remove_metainfo_if_exists(dataset_cfg):
    try:
        dataset_cfg.pop("metainfo", None)
    except Exception:
        pass


def _set_metainfo_classes(dataset_cfg, classes: List[str]):
    dataset_cfg.metainfo = dict(classes=tuple(classes))


def _build_coco_metric(ann_file: Path):
    return dict(
        type="CocoMetric",
        ann_file=str(ann_file),
        metric="bbox",
        format_only=False
    )


def infer_img_prefix_from_json(data_root: Path, ann_file: Path) -> str:
    """
    images.file_name 샘플을 보고 data_prefix.img를 자동 결정.
    - file_name이 이미 'train/xxx.jpg'처럼 하위폴더 포함이면 prefix는 ''.
    - file_name이 'xxx.jpg'면 prefix는 'train/'.
    """
    coco = load_coco(ann_file)
    imgs = coco.get("images", [])
    if not imgs:
        return "train/"

    sample = imgs[0].get("file_name", "")
    # file_name 안에 '/'가 들어가면 이미 경로가 포함된 것으로 간주
    if "/" in sample:
        return ""
    return "train/"


def patch_config_for_fold(
    cfg,
    project_root: Path,
    fold_idx: int,
    class_mode: str,
    max_epochs: int,
    batch_size: int,
    num_workers: int,
    use_amp: bool,
    grad_accum: int,
    force_img_prefix: Optional[str] = None,
):
    data_root = Path("/data/ephemeral/home/model/dataset")
    folds_dir = data_root / "folds"

    train_ann = folds_dir / f"train_fold{fold_idx}.json"
    val_ann = folds_dir / f"val_fold{fold_idx}.json"

    if not train_ann.exists() or not val_ann.exists():
        raise FileNotFoundError(f"fold json 누락: {train_ann} or {val_ann}")

    cfg.data_root = str(data_root)

    # img prefix 자동 판별
    if force_img_prefix is not None:
        img_prefix = force_img_prefix
    else:
        img_prefix = infer_img_prefix_from_json(data_root, train_ann)

    # train dataloader
    if "train_dataloader" in cfg:
        cfg.train_dataloader.batch_size = batch_size
        cfg.train_dataloader.num_workers = num_workers

        ds = cfg.train_dataloader.dataset
        ds.data_root = str(data_root)
        ds.ann_file = str(train_ann)
        ds.data_prefix = dict(img=img_prefix)

        if class_mode == "none":
            _remove_metainfo_if_exists(ds)
        elif class_mode == "auto":
            classes = load_classes_from_coco(train_ann)
            _set_metainfo_classes(ds, classes)
        else:
            raise ValueError("class_mode는 none/auto만 허용")

        # 빈 GT 필터가 데이터와 충돌할 가능성 대비
        try:
            if hasattr(ds, "filter_cfg"):
                ds.filter_cfg = dict(filter_empty_gt=False)
        except Exception:
            pass

    # val dataloader
    if "val_dataloader" in cfg:
        cfg.val_dataloader.batch_size = 1
        cfg.val_dataloader.num_workers = max(1, num_workers // 2)

        ds = cfg.val_dataloader.dataset
        ds.data_root = str(data_root)
        ds.ann_file = str(val_ann)
        ds.data_prefix = dict(img=img_prefix)

        if class_mode == "none":
            _remove_metainfo_if_exists(ds)
        elif class_mode == "auto":
            classes = load_classes_from_coco(val_ann)
            _set_metainfo_classes(ds, classes)

        try:
            if hasattr(ds, "filter_cfg"):
                ds.filter_cfg = dict(filter_empty_gt=False)
        except Exception:
            pass

    # num_classes 동기화
    classes = load_classes_from_coco(train_ann)
    num_classes = len(classes)

    if "model" in cfg:
        model = cfg.model
        if hasattr(model, "bbox_head") and isinstance(model.bbox_head, dict):
            if "num_classes" in model.bbox_head:
                model.bbox_head["num_classes"] = num_classes
        if hasattr(model, "bbox_head") and isinstance(model.bbox_head, list):
            for h in model.bbox_head:
                if isinstance(h, dict) and "num_classes" in h:
                    h["num_classes"] = num_classes

    # 10e 루프
    cfg.train_cfg = dict(type="EpochBasedTrainLoop", max_epochs=max_epochs, val_interval=1)

    # checkpoint
    cfg.default_hooks = cfg.get("default_hooks", {})
    cfg.default_hooks["checkpoint"] = dict(
        type="CheckpointHook",
        interval=1,
        max_keep_ckpts=2,
        save_best="auto"
    )

    # scheduler
    cfg.param_scheduler = [
        dict(type="LinearLR", start_factor=0.001, by_epoch=False, begin=0, end=500),
        dict(type="CosineAnnealingLR", eta_min=0.0, by_epoch=True, begin=0, end=max_epochs)
    ]

    # evaluator fold val 고정
    new_val_metric = _build_coco_metric(val_ann)
    cfg.val_evaluator = [new_val_metric] if isinstance(cfg.get("val_evaluator", None), list) else new_val_metric
    if "test_evaluator" in cfg:
        cfg.test_evaluator = [new_val_metric] if isinstance(cfg.test_evaluator, list) else new_val_metric

    # optim_wrapper 안전 구성
    # 기존 optimizer 설정을 잃지 않도록 deepcopy
    base_optim = cfg.get("optim_wrapper", None)
    base_optim_dict = copy.deepcopy(base_optim) if base_optim is not None else {}

    if use_amp:
        # DETA/DDQ 계열은 fp16 NMS dtype 이슈가 있으므로 기본적으로 비추천
        # 그래도 켜고 싶다면 optimizer를 확실히 유지
        optimizer_cfg = base_optim_dict.get("optimizer", None)
        clip_cfg = base_optim_dict.get("clip_grad", None)
        cfg.optim_wrapper = dict(
            type="AmpOptimWrapper",
            optimizer=optimizer_cfg,
            clip_grad=clip_cfg,
            loss_scale="dynamic",
            accumulative_counts=grad_accum
        )
    else:
        if isinstance(base_optim_dict, dict):
            base_optim_dict["accumulative_counts"] = grad_accum
            base_optim_dict.pop("type", None)  # 혹시 꼬임 방지
        cfg.optim_wrapper = dict(
            type="OptimWrapper",
            **base_optim_dict
        )

    # work_dir
    work_root = project_root / "work_dirs"
    tag = "classes_none" if class_mode == "none" else "classes_auto"
    cfg.work_dir = str(work_root / f"deta_10e_4fold_{tag}" / f"fold{fold_idx}")

    return cfg


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--fold", type=int, default=0, help="0~3")
    parser.add_argument("--class-mode", type=str, default="auto", choices=["none", "auto"])
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=2)

    # AMP는 기본 OFF
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--no-amp", action="store_true")
    parser.add_argument("--grad-accum", type=int, default=1)

    # 경로 문제를 강제로 제어하고 싶을 때
    parser.add_argument("--force-img-prefix", type=str, default=None,
                        help="예: '' 또는 'train/'")

    args = parser.parse_args()

    use_amp = False
    if args.amp:
        use_amp = True
    if args.no_amp:
        use_amp = False

    project_root = Path("/data/ephemeral/home/model/baseline")
    ensure_mmdet_importable(project_root)

    from mmengine.config import Config
    from mmengine.runner import Runner

    base_cfg_path = find_base_config(project_root)
    cfg = Config.fromfile(str(base_cfg_path))

    cfg.randomness = dict(seed=args.seed, deterministic=False)

    cfg = patch_config_for_fold(
        cfg=cfg,
        project_root=project_root,
        fold_idx=args.fold,
        class_mode=args.class_mode,
        max_epochs=args.epochs,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        use_amp=use_amp,
        grad_accum=args.grad_accum,
        force_img_prefix=args.force_img_prefix
    )

    runner = Runner.from_cfg(cfg)
    runner.train()


if __name__ == "__main__":
    main()
