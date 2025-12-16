# make_4fold_coco.py
# 목적:
# - baseline/dataset/train.json 을 읽어서
# - 이미지 단위로 4-fold COCO split json을 생성
# - 결과는 baseline/dataset/folds/train_fold{i}.json, val_fold{i}.json 형태로 저장

import json
import random
from pathlib import Path
from typing import Dict, List, Tuple

def load_coco(path: Path) -> Dict:
    # COCO json을 안전하게 읽는다.
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)

def save_coco(obj: Dict, path: Path) -> None:
    # COCO json을 utf-8로 저장한다.
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)

def split_image_ids(image_ids: List[int], n_folds: int, seed: int) -> List[List[int]]:
    # 이미지 id 리스트를 섞어서 n_folds로 균등 분할한다.
    rng = random.Random(seed)
    ids = image_ids[:]
    rng.shuffle(ids)

    folds = [[] for _ in range(n_folds)]
    for i, img_id in enumerate(ids):
        folds[i % n_folds].append(img_id)

    return folds

def build_subset_coco(coco: Dict, keep_image_ids: set) -> Dict:
    # 원본 COCO에서 keep_image_ids에 해당하는 이미지와 annotation만 골라낸다.
    images = [img for img in coco["images"] if img["id"] in keep_image_ids]
    annotations = [ann for ann in coco["annotations"] if ann["image_id"] in keep_image_ids]

    # categories, info, licenses 등은 그대로 유지
    out = {
        "info": coco.get("info", {}),
        "licenses": coco.get("licenses", []),
        "images": images,
        "annotations": annotations,
        "categories": coco.get("categories", []),
    }
    return out

def main():
    # ------------------------------------------------------------
    # 1) 경로를 네 구조에 딱 맞게 고정
    # ------------------------------------------------------------
    PROJECT_ROOT = Path("/data/ephemeral/home/model")
    DATA_ROOT = PROJECT_ROOT / "dataset"
    TRAIN_JSON = DATA_ROOT / "train.json"

    # ------------------------------------------------------------
    # 2) 설정 값
    # ------------------------------------------------------------
    N_FOLDS = 4
    SEED = 42

    # ------------------------------------------------------------
    # 3) 로드
    # ------------------------------------------------------------
    if not TRAIN_JSON.exists():
        raise FileNotFoundError(f"train.json 경로가 없습니다: {TRAIN_JSON}")

    coco = load_coco(TRAIN_JSON)

    # 이미지 id 수집
    image_ids = [img["id"] for img in coco["images"]]
    if len(image_ids) == 0:
        raise ValueError("train.json에 images가 비어있습니다.")

    # ------------------------------------------------------------
    # 4) fold 분할
    # ------------------------------------------------------------
    folds = split_image_ids(image_ids, N_FOLDS, SEED)

    out_dir = DATA_ROOT / "folds"
    out_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------
    # 5) fold별 train/val json 생성
    # ------------------------------------------------------------
    for i in range(N_FOLDS):
        val_ids = set(folds[i])
        train_ids = set([x for j in range(N_FOLDS) if j != i for x in folds[j]])

        train_coco = build_subset_coco(coco, train_ids)
        val_coco = build_subset_coco(coco, val_ids)

        train_path = out_dir / f"train_fold{i}.json"
        val_path = out_dir / f"val_fold{i}.json"

        save_coco(train_coco, train_path)
        save_coco(val_coco, val_path)

        print(f"[fold {i}]")
        print(f"  train images: {len(train_coco['images'])}, anns: {len(train_coco['annotations'])}")
        print(f"  val   images: {len(val_coco['images'])}, anns: {len(val_coco['annotations'])}")
        print(f"  saved: {train_path}")
        print(f"         {val_path}")

if __name__ == "__main__":
    main()
