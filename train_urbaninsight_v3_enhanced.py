#!/usr/bin/env python3
"""Enhanced UrbanInsight v3 multi-city training pipeline."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import shutil
import subprocess
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, MutableMapping, Sequence, Tuple


CITY_NAMES = ("Vegas", "Paris", "Shanghai", "Khartoum", "Rio")


@dataclass(frozen=True)
class TrainConfig:
    train_ratio: float = 0.70
    val_ratio: float = 0.15
    test_ratio: float = 0.15
    epochs: int = 50
    freeze_backbone_epochs: int = 3
    freeze_layers: int = 10
    imgsz: int = 640
    batch: int = 8
    lr0: float = 0.0003
    lrf: float = 0.01
    cos_lr: bool = True
    optimizer: str = "AdamW"
    dropout: float = 0.1
    weight_decay: float = 0.0005
    mosaic: float = 0.7
    copy_paste: float = 0.1
    close_mosaic: int = 40
    save_period: int = 5


def stable_city_seed(city: str, base_seed: int = 2024) -> int:
    digest = hashlib.sha256(f"{city.lower()}::{base_seed}".encode("utf-8")).hexdigest()
    return int(digest[:8], 16)


def split_city_tiles(tile_ids: Sequence[str], city: str, cfg: TrainConfig) -> Dict[str, List[str]]:
    if abs((cfg.train_ratio + cfg.val_ratio + cfg.test_ratio) - 1.0) > 1e-9:
        raise ValueError("train/val/test ratios must sum to 1.0")

    ids = list(tile_ids)
    rng = random.Random(stable_city_seed(city))
    rng.shuffle(ids)

    n = len(ids)
    n_train = int(n * cfg.train_ratio)
    n_val = int(n * cfg.val_ratio)
    n_test = n - n_train - n_val

    train_ids = ids[:n_train]
    val_ids = ids[n_train : n_train + n_val]
    test_ids = ids[n_train + n_val : n_train + n_val + n_test]

    return {"train": train_ids, "val": val_ids, "test": test_ids}


def verify_zero_data_leakage(city_splits: Mapping[str, Mapping[str, Sequence[str]]]) -> None:
    for city, splits in city_splits.items():
        train_set = set(splits.get("train", []))
        val_set = set(splits.get("val", []))
        test_set = set(splits.get("test", []))

        overlaps = {
            "train∩val": train_set & val_set,
            "train∩test": train_set & test_set,
            "val∩test": val_set & test_set,
        }
        non_empty = {k: sorted(v)[:5] for k, v in overlaps.items() if v}
        if non_empty:
            raise ValueError(f"Data leakage detected for city '{city}': {non_empty}")


def hash_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def detect_hash_leakage(split_to_files: Mapping[str, Iterable[Path]]) -> Dict[str, Dict[str, List[str]]]:
    hash_to_split_files: Dict[str, Dict[str, List[str]]] = {}
    for split, files in split_to_files.items():
        for file_path in files:
            digest = hash_file(Path(file_path))
            bucket = hash_to_split_files.setdefault(digest, {})
            bucket.setdefault(split, []).append(str(file_path))

    leakage = {
        digest: details
        for digest, details in hash_to_split_files.items()
        if len(details.keys()) > 1
    }
    return leakage


def find_latest_checkpoint(weights_dir: Path) -> Path | None:
    if not weights_dir.exists():
        return None
    ranked: List[Tuple[int, Path]] = []
    for candidate in weights_dir.glob("epoch*.pt"):
        suffix = candidate.stem.replace("epoch", "")
        if suffix.isdigit():
            ranked.append((int(suffix), candidate))
    if ranked:
        return sorted(ranked, key=lambda x: x[0])[-1][1]

    for fallback in (weights_dir / "last.pt", weights_dir / "best.pt"):
        if fallback.exists():
            return fallback
    return None


def current_git_commit(repo_dir: Path) -> str:
    try:
        out = subprocess.check_output(
            ["git", "-C", str(repo_dir), "rev-parse", "HEAD"],
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
        return out
    except Exception:
        return "unknown"


def write_json(path: Path, payload: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def build_train_kwargs(cfg: TrainConfig, data_yaml: str, project_dir: Path, run_name: str, *, freeze: int = 0) -> Dict[str, object]:
    return {
        "data": data_yaml,
        "epochs": cfg.epochs,
        "imgsz": cfg.imgsz,
        "batch": cfg.batch,
        "optimizer": cfg.optimizer,
        "lr0": cfg.lr0,
        "lrf": cfg.lrf,
        "cos_lr": cfg.cos_lr,
        "dropout": cfg.dropout,
        "weight_decay": cfg.weight_decay,
        "mosaic": cfg.mosaic,
        "copy_paste": cfg.copy_paste,
        "close_mosaic": cfg.close_mosaic,
        "save": True,
        "save_period": cfg.save_period,
        "freeze": freeze,
        "project": str(project_dir),
        "name": run_name,
        "exist_ok": True,
    }


def run_training(args: argparse.Namespace) -> None:
    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise RuntimeError("ultralytics is required. Install with `pip install ultralytics`.") from exc

    cfg = TrainConfig()
    out_dir = Path(args.output_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    run_meta: MutableMapping[str, object] = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "git_commit": current_git_commit(Path(args.repo_dir).resolve()),
        "config": asdict(cfg),
        "cities": list(CITY_NAMES),
    }

    phase1_model = YOLO(args.model)
    warmup_kwargs = build_train_kwargs(
        cfg,
        data_yaml=args.data_yaml,
        project_dir=out_dir,
        run_name="urbaninsight_v3_warmup",
        freeze=cfg.freeze_layers,
    )
    warmup_kwargs["epochs"] = cfg.freeze_backbone_epochs
    phase1_model.train(**warmup_kwargs)

    warmup_last = out_dir / "urbaninsight_v3_warmup" / "weights" / "last.pt"
    phase2_start = warmup_last if warmup_last.exists() else Path(args.model)

    remaining_epochs = cfg.epochs - cfg.freeze_backbone_epochs
    if remaining_epochs <= 0:
        raise ValueError("epochs must be greater than freeze_backbone_epochs")

    phase2_model = YOLO(str(phase2_start))
    main_kwargs = build_train_kwargs(
        cfg,
        data_yaml=args.data_yaml,
        project_dir=out_dir,
        run_name="urbaninsight_v3_main",
        freeze=0,
    )
    main_kwargs["epochs"] = remaining_epochs
    phase2_model.train(**main_kwargs)

    weights_dir = out_dir / "urbaninsight_v3_main" / "weights"
    best_pt = weights_dir / "best.pt"
    last_pt = find_latest_checkpoint(weights_dir)

    if not best_pt.exists() and last_pt is not None:
        best_pt = last_pt

    if not best_pt.exists():
        raise FileNotFoundError("No trained checkpoint found in main training run")

    final_merged = out_dir / "final_merged_model.pt"
    shutil.copy2(best_pt, final_merged)

    per_city_metrics: Dict[str, Dict[str, float]] = {}
    for city, city_yaml in json.loads(Path(args.city_val_yamls_json).read_text(encoding="utf-8")).items():
        metrics = phase2_model.val(data=city_yaml)
        summary = {
            "precision": float(metrics.box.mp),
            "recall": float(metrics.box.mr),
            "map50": float(metrics.box.map50),
            "map50_95": float(metrics.box.map),
            "seg_map50": float(metrics.seg.map50),
            "seg_map50_95": float(metrics.seg.map),
        }
        per_city_metrics[city] = summary
        shutil.copy2(best_pt, out_dir / f"best_{city}.pt")

    run_meta["per_city_metrics"] = per_city_metrics
    run_meta["final_model"] = str(final_merged)
    run_meta["latest_checkpoint"] = str(last_pt) if last_pt else None
    write_json(out_dir / "training_run_metadata.json", run_meta)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Enhanced UrbanInsight v3 training pipeline")
    parser.add_argument("--data-yaml", required=True, help="Combined multi-city YOLO data YAML")
    parser.add_argument("--city-val-yamls-json", required=True, help="JSON map: city -> YAML file for per-city validation")
    parser.add_argument("--model", required=True, help="Path/model id for YOLOv8-seg pretrained weights")
    parser.add_argument("--output-dir", default="./runs", help="Output directory for checkpoints and logs")
    parser.add_argument("--repo-dir", default=".", help="Repository root to capture git commit")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_training(args)


if __name__ == "__main__":
    main()
