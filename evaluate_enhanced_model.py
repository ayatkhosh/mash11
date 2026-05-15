#!/usr/bin/env python3
"""Evaluation suite for enhanced UrbanInsight v3 model."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Sequence

from pipeline_utils import hash_file


PAPER_BENCHMARKS = {"precision": 0.944, "recall": 0.892, "map50": 0.945}


@dataclass(frozen=True)
class DatasetEvalResult:
    dataset: str
    precision: float
    recall: float
    map50: float
    map50_95: float
    seg_map50: float
    seg_map50_95: float


def verify_city_split_isolation(split_spec: Mapping[str, Mapping[str, Sequence[str]]]) -> Dict[str, Dict[str, List[str]]]:
    leakage: Dict[str, Dict[str, List[str]]] = {}
    for city, splits in split_spec.items():
        train_set = set(splits.get("train", []))
        val_set = set(splits.get("val", []))
        test_set = set(splits.get("test", []))
        overlaps = {
            "train_val": sorted(train_set & val_set),
            "train_test": sorted(train_set & test_set),
            "val_test": sorted(val_set & test_set),
        }
        overlaps = {k: v for k, v in overlaps.items() if v}
        if overlaps:
            leakage[city] = overlaps
    return leakage


def detect_hash_leakage(split_to_files: Mapping[str, Iterable[Path]]) -> Dict[str, Dict[str, List[str]]]:
    digest_map: Dict[str, Dict[str, List[str]]] = {}
    for split, paths in split_to_files.items():
        for p in paths:
            d = hash_file(Path(p))
            digest_map.setdefault(d, {}).setdefault(split, []).append(str(p))
    return {d: v for d, v in digest_map.items() if len(v.keys()) > 1}


def yolo_val(model_path: str, data_yaml: str) -> DatasetEvalResult:
    from ultralytics import YOLO

    model = YOLO(model_path)
    metrics = model.val(data=data_yaml)
    return DatasetEvalResult(
        dataset=data_yaml,
        precision=float(metrics.box.mp),
        recall=float(metrics.box.mr),
        map50=float(metrics.box.map50),
        map50_95=float(metrics.box.map),
        seg_map50=float(metrics.seg.map50),
        seg_map50_95=float(metrics.seg.map),
    )


def evaluate_fmow_external(model_path: str, fmow_images_dir: Path, conf: float) -> Dict[str, float]:
    from ultralytics import YOLO

    model = YOLO(model_path)
    image_paths = sorted(
        [*fmow_images_dir.glob("*.png"), *fmow_images_dir.glob("*.jpg"), *fmow_images_dir.glob("*.jpeg"), *fmow_images_dir.glob("*.tif")]
    )
    if not image_paths:
        return {"images": 0, "avg_detections": 0.0, "avg_confidence": 0.0}

    detections = 0
    conf_sum = 0.0
    conf_count = 0
    for image_path in image_paths:
        result = model.predict(str(image_path), conf=conf, verbose=False)[0]
        if result.boxes is None:
            continue
        detections += len(result.boxes)
        for c in result.boxes.conf.cpu().numpy().tolist():
            conf_sum += float(c)
            conf_count += 1

    return {
        "images": len(image_paths),
        "avg_detections": float(detections / len(image_paths)),
        "avg_confidence": float(conf_sum / conf_count) if conf_count else 0.0,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate enhanced UrbanInsight model")
    parser.add_argument("--model", required=True, help="Path to trained model")
    parser.add_argument("--city-yamls-json", required=True, help="JSON mapping city->YOLO dataset yaml for per-city test")
    parser.add_argument("--spacenet-merged-yaml", required=True, help="Merged SpaceNet test yaml")
    parser.add_argument("--fmow-images-dir", required=True, help="FMOW image directory for external testing")
    parser.add_argument("--split-spec-json", required=True, help="Split spec JSON (city -> {train,val,test})")
    parser.add_argument("--output", default="./evaluation_report.json", help="Output report path")
    parser.add_argument("--conf", type=float, default=0.25, help="External inference confidence")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    city_yamls = json.loads(Path(args.city_yamls_json).read_text(encoding="utf-8"))
    split_spec = json.loads(Path(args.split_spec_json).read_text(encoding="utf-8"))

    city_leakage = verify_city_split_isolation(split_spec)

    per_city = {}
    for city, yaml_path in city_yamls.items():
        per_city[city] = yolo_val(args.model, yaml_path).__dict__

    spacenet_merged = yolo_val(args.model, args.spacenet_merged_yaml).__dict__
    fmow_summary = evaluate_fmow_external(args.model, Path(args.fmow_images_dir), args.conf)

    benchmark_gaps = {
        key: float(spacenet_merged.get(key, 0.0) - PAPER_BENCHMARKS[key])
        for key in PAPER_BENCHMARKS
    }

    report = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "paper_benchmarks": PAPER_BENCHMARKS,
        "spacenet": {"merged": spacenet_merged, "per_city": per_city},
        "fmow_external": fmow_summary,
        "benchmark_gaps": benchmark_gaps,
        "data_leakage": {
            "city_split_overlap": city_leakage,
            "status": "pass" if not city_leakage else "fail",
        },
    }

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    main()
