from __future__ import annotations

from pathlib import Path

import numpy as np
import rasterio
from rasterio.warp import Resampling, reproject


def _read_mask(path: Path) -> tuple[np.ndarray, dict]:
    with rasterio.open(path) as src:
        arr = src.read(1)
        meta = src.meta.copy()
        nodata = src.nodata
    mask = np.isfinite(arr)
    if nodata is not None:
        mask &= arr != nodata
    binary = np.where(mask & (arr > 0), 1, 0).astype(np.uint8)
    return binary, meta


def align_observed_mask_to_predicted_grid(observed_path: Path, predicted_meta: dict) -> np.ndarray:
    with rasterio.open(observed_path) as src:
        dst = np.zeros((predicted_meta["height"], predicted_meta["width"]), dtype="uint8")
        reproject(
            source=rasterio.band(src, 1),
            destination=dst,
            src_transform=src.transform,
            src_crs=src.crs,
            dst_transform=predicted_meta["transform"],
            dst_crs=predicted_meta["crs"],
            src_nodata=src.nodata,
            dst_nodata=0,
            resampling=Resampling.nearest,
        )
    return np.where(dst > 0, 1, 0).astype(np.uint8)


def compute_binary_metrics(predicted: np.ndarray, observed: np.ndarray) -> dict[str, float]:
    pred = predicted.astype(bool)
    obs = observed.astype(bool)
    tp = int(np.logical_and(pred, obs).sum())
    fp = int(np.logical_and(pred, ~obs).sum())
    fn = int(np.logical_and(~pred, obs).sum())
    tn = int(np.logical_and(~pred, ~obs).sum())

    iou = tp / (tp + fp + fn) if (tp + fp + fn) else 0.0
    dice = (2 * tp) / (2 * tp + fp + fn) if (2 * tp + fp + fn) else 0.0
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0

    return {
        "tp": float(tp),
        "fp": float(fp),
        "fn": float(fn),
        "tn": float(tn),
        "iou": float(iou),
        "dice": float(dice),
        "precision": float(precision),
        "recall": float(recall),
    }


def compute_raster_metrics(predicted_path: Path, observed_path: Path) -> dict[str, float]:
    predicted, predicted_meta = _read_mask(predicted_path)
    observed = align_observed_mask_to_predicted_grid(observed_path, predicted_meta)
    return compute_binary_metrics(predicted, observed)
