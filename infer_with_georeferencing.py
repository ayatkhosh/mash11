#!/usr/bin/env python3
"""Inference pipeline with georeferencing and JSON export."""

from __future__ import annotations

import argparse
import base64
import json
import warnings
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

RESIDENTIAL_AREA_THRESHOLD_M2 = 600.0


def encode_mask(mask) -> str:
    import cv2
    import numpy as np

    ok, encoded = cv2.imencode(".png", (mask > 0).astype(np.uint8) * 255)
    if not ok:
        raise RuntimeError("Failed to encode segmentation mask")
    return base64.b64encode(encoded.tobytes()).decode("utf-8")


def classify_building_type(footprint_area_m2: float) -> str:
    return "residential" if footprint_area_m2 < RESIDENTIAL_AREA_THRESHOLD_M2 else "non-residential"


def contour_polygon(mask: "np.ndarray") -> List[Tuple[float, float]]:
    import cv2
    import numpy as np

    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return []
    contour = max(contours, key=cv2.contourArea)
    return [(float(p[0][0]), float(p[0][1])) for p in contour]


def geo_coords_from_pixels(pixel_polygon: Sequence[Tuple[float, float]], transform, src_crs) -> List[List[float]]:
    if not pixel_polygon:
        return []

    from pyproj import Transformer
    from rasterio.transform import xy

    to_wgs84 = None
    if src_crs and str(src_crs).upper() != "EPSG:4326":
        to_wgs84 = Transformer.from_crs(src_crs, "EPSG:4326", always_xy=True)

    coords: List[List[float]] = []
    for x, y in pixel_polygon:
        lon, lat = xy(transform, y, x)
        if to_wgs84 is not None:
            lon, lat = to_wgs84.transform(lon, lat)
        coords.append([float(lon), float(lat)])

    if coords and coords[0] != coords[-1]:
        coords.append(coords[0])
    return coords


def run_inference(image_path: Path, model_path: str, conf: float, output_dir: Path) -> Dict[str, object]:
    from rasterio.transform import Affine
    import cv2
    import numpy as np

    try:
        import rasterio
    except ImportError as exc:
        raise RuntimeError("rasterio is required. Install with `pip install rasterio`.") from exc

    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise RuntimeError("ultralytics is required. Install with `pip install ultralytics`.") from exc

    output_dir.mkdir(parents=True, exist_ok=True)

    transform = Affine.identity()
    src_crs = "EPSG:4326"
    gsd = 1.0

    image_rgb = None
    try:
        with rasterio.open(image_path) as src:
            src_crs = src.crs if src.crs else "EPSG:4326"
            transform = src.transform
            gsd = float((abs(transform.a) + abs(transform.e)) / 2.0)
            bands = src.read()
            if bands.shape[0] >= 3:
                image_rgb = np.transpose(bands[:3], (1, 2, 0))
            else:
                gray = bands[0]
                image_rgb = np.stack([gray, gray, gray], axis=-1)
            image_rgb = np.clip(image_rgb, 0, 255).astype(np.uint8)
    except Exception:
        image_bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image_bgr is None:
            raise FileNotFoundError(f"Unable to read image: {image_path}")
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

    model = YOLO(model_path)
    result = model.predict(str(image_path), conf=conf, verbose=False)[0]

    boxes = result.boxes
    masks = result.masks.data.cpu().numpy() if result.masks is not None else None

    vis = image_rgb.copy()
    buildings: List[Dict[str, object]] = []

    for idx in range(len(boxes) if boxes is not None else 0):
        box_xywh = boxes.xywh[idx].cpu().numpy().tolist()
        confidence = float(boxes.conf[idx].cpu().item())
        if masks is None or idx >= len(masks):
            warnings.warn(
                f\"Missing segmentation mask for detection index {idx}; using an empty fallback mask.\", RuntimeWarning
            )
            mask = np.zeros(image_rgb.shape[:2], dtype=np.uint8)
        else:
            mask = masks[idx]
        if mask.shape != image_rgb.shape[:2]:
            mask = cv2.resize(mask, (image_rgb.shape[1], image_rgb.shape[0]), interpolation=cv2.INTER_NEAREST)
        binary = (mask > 0.5).astype(np.uint8)

        area_m2 = float(binary.sum() * (gsd**2))
        pixel_polygon = contour_polygon(binary)
        geographic_polygon = geo_coords_from_pixels(pixel_polygon, transform, src_crs)

        color = np.array([0, 255, 0], dtype=np.uint8)
        vis[binary > 0] = (0.6 * vis[binary > 0] + 0.4 * color).astype(np.uint8)

        x, y, w, h = [float(v) for v in box_xywh]
        cv2.rectangle(
            vis,
            (int(x - w / 2), int(y - h / 2)),
            (int(x + w / 2), int(y + h / 2)),
            (255, 0, 0),
            2,
        )

        buildings.append(
            {
                "id": f"building_{idx+1:03d}",
                "type": classify_building_type(area_m2),
                "bounding_box": {"x": x, "y": y, "width": w, "height": h},
                "segmentation_mask": encode_mask(binary),
                "footprint_area_m2": area_m2,
                "confidence": confidence,
                "polygon_coordinates": geographic_polygon,
            }
        )

    payload = {
        "image_name": image_path.name,
        "total_buildings": len(buildings),
        "gsd": gsd,
        "buildings": buildings,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "coordinate_system": "EPSG:4326",
    }

    json_path = output_dir / f"{image_path.stem}_buildings.json"
    json_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")

    vis_bgr = cv2.cvtColor(vis, cv2.COLOR_RGB2BGR)
    cv2.imwrite(str(output_dir / f"{image_path.stem}_overlay.png"), vis_bgr)

    geojson = {
        "type": "FeatureCollection",
        "features": [
            {
                "type": "Feature",
                "properties": {
                    "id": b["id"],
                    "type": b["type"],
                    "footprint_area_m2": b["footprint_area_m2"],
                    "confidence": b["confidence"],
                },
                "geometry": {"type": "Polygon", "coordinates": [b["polygon_coordinates"]]},
            }
            for b in buildings
            if b["polygon_coordinates"]
        ],
    }
    (output_dir / f"{image_path.stem}_buildings.geojson").write_text(
        json.dumps(geojson, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="YOLOv8-seg inference with georeferencing")
    parser.add_argument("--model", required=True, help="Trained YOLOv8-seg weights")
    parser.add_argument("--image", required=True, help="Input image path")
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold")
    parser.add_argument("--output-dir", default="./inference_output", help="Output folder")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_inference(Path(args.image).resolve(), args.model, args.conf, Path(args.output_dir).resolve())


if __name__ == "__main__":
    main()
