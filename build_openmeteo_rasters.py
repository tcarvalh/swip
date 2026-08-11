from __future__ import annotations

import datetime
import argparse
import math
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import h3.api.basic_int as h3
import numpy as np
from pyproj import Transformer
import rasterio
import requests
from rasterio.features import rasterize, shapes
from rasterio.transform import from_origin, xy
from rasterio.warp import Resampling, reproject, transform_geom
from shapely.geometry import MultiPolygon, Polygon, mapping, shape
from shapely.ops import unary_union

APP_DIR = Path(__file__).resolve().parent
BP_PATH = APP_DIR / "BP" / "Tennessee_BP.tif"
OUT_U_DIR = APP_DIR / "openmeteo_windu_rast"
OUT_V_DIR = APP_DIR / "openmeteo_windv_rast"
OUT_RH_DIR = APP_DIR / "openmeteo_rh_rast"
FORECAST_API_URL = "https://api.open-meteo.com/v1/forecast"
ARCHIVE_API_URL = "https://archive-api.open-meteo.com/v1/archive"
NODATA = np.float32(-3.4028235e38)
ProgressCallback = Callable[[int, str], None]

HTTP_SESSION = requests.Session()
HTTP_SESSION.trust_env = False


@dataclass
class SamplePoint:
    lat: float
    lon: float
    wind_speed_ms: float
    wind_direction_from_deg: float
    wind_u_ms: float
    wind_v_ms: float
    relative_humidity_pct: float


def _debug(msg: str) -> None:
    print(msg, flush=True)


def _report(msg: str, progress: int | None = None, progress_callback: ProgressCallback | None = None) -> None:
    _debug(msg)
    if progress is not None and progress_callback is not None:
        pct = max(0, min(100, int(progress)))
        progress_callback(pct, msg)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create Open-Meteo raster overlays from H3 centroid samples."
    )
    parser.add_argument("--h3-resolution", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=48)
    parser.add_argument("--idw-neighbors", type=int, default=8)
    parser.add_argument("--idw-power", type=float, default=2.0)
    parser.add_argument("--interp-chunk-size", type=int, default=20000)
    parser.add_argument("--timeout-sec", type=int, default=30)
    parser.add_argument("--timezone", default="UTC")
    parser.add_argument("--wind-speed-unit", default="ms")
    parser.add_argument("--u-out", default=str(OUT_U_DIR / "openmeteo_wind_u_r5_now.tif"))
    parser.add_argument("--v-out", default=str(OUT_V_DIR / "openmeteo_wind_v_r5_now.tif"))
    parser.add_argument("--rh-out", default=str(OUT_RH_DIR / "openmeteo_rh_r5_now.tif"))
    parser.add_argument("--target-hour", default="")
    parser.add_argument("--use-current", action="store_true")
    parser.add_argument("--request-retries", type=int, default=4)
    return parser.parse_args()


def _iter_polygons(geom: Polygon | MultiPolygon):
    if isinstance(geom, Polygon):
        yield geom
        return
    if isinstance(geom, MultiPolygon):
        for poly in geom.geoms:
            yield poly
        return
    raise ValueError(f"Unsupported geometry type: {geom.geom_type}")


def _valid_bp_geometry_wgs84(bp_path: Path) -> Polygon | MultiPolygon:
    with rasterio.open(bp_path) as src:
        arr = src.read(1).astype("float32")
        valid = np.isfinite(arr)
        if src.nodata is not None:
            valid &= arr != src.nodata
        valid &= arr > 0

        if not np.any(valid):
            raise ValueError("BP raster has no valid area.")

        polys = []
        for geom, val in shapes(valid.astype("uint8"), mask=valid, transform=src.transform):
            if int(val) == 1:
                polys.append(shape(geom))
        merged = unary_union(polys)
        if src.crs is None:
            raise ValueError("BP raster CRS is missing.")
        geom_wgs84 = shape(rasterio.warp.transform_geom(src.crs, "EPSG:4326", mapping(merged)))
        return geom_wgs84


def _h3_cells_from_geometry(geom: Polygon | MultiPolygon, resolution: int) -> list[int]:
    cells: set[int] = set()
    for poly in _iter_polygons(geom):
        if poly.is_empty:
            continue
        cells |= set(h3.geo_to_cells(mapping(poly), resolution))
    return sorted(cells)


def _batched(seq: list[int], size: int):
    for i in range(0, len(seq), size):
        yield seq[i : i + size]


def wind_from_direction_to_uv(speed_ms: float, wind_from_deg: float) -> tuple[float, float]:
    # Open-Meteo reports the direction the wind is coming from, clockwise from north.
    # Convert that meteorological convention into east/north velocity components that
    # point toward the direction the wind is blowing to.
    radians = math.radians(wind_from_deg)
    u_ms = -speed_ms * math.sin(radians)
    v_ms = -speed_ms * math.cos(radians)
    return u_ms, v_ms


def _normalize_target_hour(target_hour: str | None) -> str | None:
    raw = (target_hour or "").strip()
    if not raw:
        return None
    try:
        parsed = datetime.datetime.strptime(raw, "%Y-%m-%dT%H:%M")
    except ValueError as exc:
        raise ValueError("target_hour must use YYYY-MM-DDTHH:00 format.") from exc
    if parsed.minute != 0:
        raise ValueError("target_hour must be specified to the hour.")
    return parsed.strftime("%Y-%m-%dT%H:%M")


def _extract_sample(entry: dict, target_hour: str | None = None) -> tuple[float, float, float, float, float] | None:
    if target_hour is None:
        current = entry.get("current") or {}
        wind = current.get("wind_speed_10m")
        wind_from_deg = current.get("wind_direction_10m")
        rh = current.get("relative_humidity_2m")
    else:
        hourly = entry.get("hourly") or {}
        times = hourly.get("time") or []
        wind_series = hourly.get("wind_speed_10m") or []
        dir_series = hourly.get("wind_direction_10m") or []
        rh_series = hourly.get("relative_humidity_2m") or []
        if not (times and wind_series and dir_series and rh_series):
            return None
        try:
            hour_idx = list(times).index(target_hour)
        except ValueError:
            hour_idx = 0 if len(times) == 1 else -1
        if hour_idx < 0:
            return None
        try:
            wind = wind_series[hour_idx]
            wind_from_deg = dir_series[hour_idx]
            rh = rh_series[hour_idx]
        except (IndexError, TypeError):
            return None
    if wind is None or wind_from_deg is None or rh is None:
        return None
    try:
        wind_v = float(wind)
        wind_from_deg_v = float(wind_from_deg)
        rh_v = float(rh)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(wind_v) or not math.isfinite(wind_from_deg_v) or not math.isfinite(rh_v):
        return None
    u_ms, v_ms = wind_from_direction_to_uv(wind_v, wind_from_deg_v)
    return wind_v, wind_from_deg_v, u_ms, v_ms, rh_v


def fetch_openmeteo_samples(
    cells: list[int],
    batch_size: int,
    timeout_sec: int,
    timezone: str,
    wind_speed_unit: str,
    target_hour: str | None = None,
    request_retries: int = 4,
    progress_callback: ProgressCallback | None = None,
    progress_start: int = 15,
    progress_end: int = 55,
) -> list[SamplePoint]:
    samples: list[SamplePoint] = []
    hour_note = f" for {target_hour}" if target_hour else " for current weather"
    _report(
        f"Fetching Open-Meteo data for {len(cells)} H3 cells in batches of {batch_size}{hour_note}...",
        progress_start,
        progress_callback,
    )

    total_batches = max(1, math.ceil(len(cells) / batch_size))
    for idx, batch in enumerate(_batched(cells, batch_size), start=1):
        lats_lons = [h3.cell_to_latlng(cell) for cell in batch]
        lat_csv = ",".join(f"{lat:.6f}" for lat, _ in lats_lons)
        lon_csv = ",".join(f"{lon:.6f}" for _, lon in lats_lons)
        request_url = FORECAST_API_URL
        params = {
            "latitude": lat_csv,
            "longitude": lon_csv,
            "timezone": timezone,
            "wind_speed_unit": wind_speed_unit,
        }
        if target_hour:
            request_url = ARCHIVE_API_URL
            target_date = target_hour.split("T", 1)[0]
            params["hourly"] = "wind_speed_10m,wind_direction_10m,relative_humidity_2m"
            params["start_date"] = target_date
            params["end_date"] = target_date
            params["timeformat"] = "iso8601"
        else:
            params["current"] = "wind_speed_10m,wind_direction_10m,relative_humidity_2m"
        last_err: Exception | None = None
        resp = None
        for attempt in range(1, max(1, request_retries) + 1):
            try:
                resp = HTTP_SESSION.get(request_url, params=params, timeout=timeout_sec)
                if resp.status_code == 429:
                    retry_after = resp.headers.get("Retry-After")
                    if retry_after is not None:
                        try:
                            sleep_sec = max(1.0, float(retry_after))
                        except ValueError:
                            sleep_sec = float(min(30, 2 ** attempt))
                    else:
                        sleep_sec = float(min(30, 2 ** attempt))
                    if attempt < max(1, request_retries):
                        _report(
                            f"  Batch {idx}/{total_batches}: rate limited, retrying in {sleep_sec:.1f}s (attempt {attempt}/{request_retries})",
                            int(progress_start + ((progress_end - progress_start) * (idx - 1) / total_batches)),
                            progress_callback,
                        )
                        time.sleep(sleep_sec)
                        continue
                resp.raise_for_status()
                break
            except Exception as exc:
                last_err = exc
                if attempt < max(1, request_retries):
                    sleep_sec = float(min(30, 2 ** attempt))
                    _report(
                        f"  Batch {idx}/{total_batches}: request failed, retrying in {sleep_sec:.1f}s (attempt {attempt}/{request_retries})",
                        int(progress_start + ((progress_end - progress_start) * (idx - 1) / total_batches)),
                        progress_callback,
                    )
                    time.sleep(sleep_sec)
                    continue
                raise last_err

        if resp is None:
            raise RuntimeError("Open-Meteo request failed without response.")
        payload = resp.json()
        entries = payload if isinstance(payload, list) else [payload]
        if len(entries) != len(batch):
            raise RuntimeError(
                f"Open-Meteo response count mismatch for batch {idx}: "
                f"requested={len(batch)} got={len(entries)}"
            )

        accepted = 0
        for (lat, lon), entry in zip(lats_lons, entries):
            values = _extract_sample(entry, target_hour=target_hour)
            if values is None:
                continue
            wind_v, wind_from_deg_v, u_ms, v_ms, rh_v = values
            samples.append(
                SamplePoint(
                    lat=float(lat),
                    lon=float(lon),
                    wind_speed_ms=wind_v,
                    wind_direction_from_deg=wind_from_deg_v,
                    wind_u_ms=u_ms,
                    wind_v_ms=v_ms,
                    relative_humidity_pct=rh_v,
                )
            )
            accepted += 1

        batch_progress = progress_start + ((progress_end - progress_start) * idx / total_batches)
        _report(
            f"  Batch {idx}/{total_batches}: requested={len(batch)} accepted={accepted}",
            int(batch_progress),
            progress_callback,
        )

    if not samples:
        raise RuntimeError("No usable Open-Meteo samples were returned.")
    _report(f"Collected {len(samples)} usable sample points.", progress_end, progress_callback)
    return samples


def idw_interpolate(
    sample_xy: np.ndarray,
    sample_values: np.ndarray,
    target_xy: np.ndarray,
    neighbors: int,
    power: float,
    chunk_size: int,
    chunk_callback: Callable[[float], None] | None = None,
) -> np.ndarray:
    n_samples = sample_xy.shape[0]
    if n_samples == 0:
        raise ValueError("No sample points for interpolation.")

    k = min(neighbors, n_samples)
    out = np.full(target_xy.shape[0], np.nan, dtype="float32")
    sample_x = sample_xy[:, 0]
    sample_y = sample_xy[:, 1]

    total_chunks = max(1, math.ceil(target_xy.shape[0] / chunk_size))
    chunk_idx = 0
    for start in range(0, target_xy.shape[0], chunk_size):
        end = min(start + chunk_size, target_xy.shape[0])
        chunk = target_xy[start:end]
        dx = chunk[:, None, 0] - sample_x[None, :]
        dy = chunk[:, None, 1] - sample_y[None, :]
        d2 = dx * dx + dy * dy

        near_idx = np.argpartition(d2, kth=k - 1, axis=1)[:, :k]
        near_d2 = np.take_along_axis(d2, near_idx, axis=1)
        near_vals = np.take(sample_values, near_idx)

        exact_mask = near_d2 == 0.0
        exact_rows = np.any(exact_mask, axis=1)
        if np.any(exact_rows):
            exact_cols = np.argmax(exact_mask[exact_rows], axis=1)
            out[start:end][exact_rows] = near_vals[exact_rows, exact_cols].astype("float32")

        interp_rows = ~exact_rows
        if np.any(interp_rows):
            d = np.sqrt(near_d2[interp_rows])
            w = 1.0 / np.power(np.maximum(d, 1e-12), power)
            vw = np.sum(w * near_vals[interp_rows], axis=1)
            ws = np.sum(w, axis=1)
            out[start:end][interp_rows] = (vw / ws).astype("float32")

        chunk_idx += 1
        if chunk_callback is not None:
            chunk_callback(chunk_idx / total_chunks)

    return out


def interpolate_to_bp_grid(
    samples: list[SamplePoint],
    mask_geometry_wgs84: Polygon | MultiPolygon,
    neighbors: int,
    power: float,
    chunk_size: int,
    progress_callback: ProgressCallback | None = None,
    progress_start: int = 55,
    progress_end: int = 90,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict]:
    with rasterio.open(BP_PATH) as bp_src:
        if bp_src.crs is None:
            raise ValueError("BP raster has no CRS.")
        left, bottom, right, top = bp_src.bounds
        target_crs = bp_src.crs
        target_res_x = abs(float(bp_src.transform.a))
        target_res_y = abs(float(bp_src.transform.e))

    width = max(1, int(math.ceil((right - left) / target_res_x)))
    height = max(1, int(math.ceil((top - bottom) / target_res_y)))
    target_transform = from_origin(left, top, target_res_x, target_res_y)

    meta = {
        "driver": "GTiff",
        "dtype": "float32",
        "count": 1,
        "width": width,
        "height": height,
        "crs": target_crs,
        "transform": target_transform,
        "nodata": float(NODATA),
    }

    geom_template_crs = transform_geom("EPSG:4326", target_crs, mapping(mask_geometry_wgs84), precision=6)
    area_mask = rasterize(
        [(geom_template_crs, 1)],
        out_shape=(height, width),
        transform=target_transform,
        fill=0,
        dtype="uint8",
    )
    valid = area_mask.astype(bool)

    row_idx, col_idx = np.where(valid)
    xs, ys = xy(target_transform, row_idx, col_idx, offset="center")
    target_xy = np.column_stack((np.asarray(xs, dtype="float64"), np.asarray(ys, dtype="float64")))

    transformer = Transformer.from_crs("EPSG:4326", target_crs, always_xy=True)
    sample_lon = np.array([s.lon for s in samples], dtype="float64")
    sample_lat = np.array([s.lat for s in samples], dtype="float64")
    sx, sy = transformer.transform(sample_lon, sample_lat)
    sample_xy = np.column_stack((sx, sy))

    wind_speed_values = np.array([s.wind_speed_ms for s in samples], dtype="float64")
    wind_u_values = np.array([s.wind_u_ms for s in samples], dtype="float64")
    wind_v_values = np.array([s.wind_v_ms for s in samples], dtype="float64")
    rh_values = np.array([s.relative_humidity_pct for s in samples], dtype="float64")

    progress_span = progress_end - progress_start
    speed_start = progress_start
    speed_end = progress_start + int(progress_span * 0.25)
    u_start = speed_end
    u_end = progress_start + int(progress_span * 0.5)
    v_start = u_end
    v_end = progress_start + int(progress_span * 0.75)
    rh_start = v_end
    rh_end = progress_end

    _report(f"Interpolating wind speed to {target_xy.shape[0]} target cells...", speed_start, progress_callback)
    wind_speed_interp = idw_interpolate(
        sample_xy,
        wind_speed_values,
        target_xy,
        neighbors,
        power,
        chunk_size,
        chunk_callback=(
            (
                lambda ratio: progress_callback(
                    int(speed_start + ratio * (speed_end - speed_start)),
                    "Interpolating wind speed...",
                )
            )
            if progress_callback is not None
            else None
        ),
    )

    _report(f"Interpolating wind U to {target_xy.shape[0]} target cells...", u_start, progress_callback)
    wind_u_interp = idw_interpolate(
        sample_xy,
        wind_u_values,
        target_xy,
        neighbors,
        power,
        chunk_size,
        chunk_callback=(
            (
                lambda ratio: progress_callback(
                    int(u_start + ratio * (u_end - u_start)),
                    "Interpolating wind U...",
                )
            )
            if progress_callback is not None
            else None
        ),
    )

    _report(f"Interpolating wind V to {target_xy.shape[0]} target cells...", v_start, progress_callback)
    wind_v_interp = idw_interpolate(
        sample_xy,
        wind_v_values,
        target_xy,
        neighbors,
        power,
        chunk_size,
        chunk_callback=(
            (
                lambda ratio: progress_callback(
                    int(v_start + ratio * (v_end - v_start)),
                    "Interpolating wind V...",
                )
            )
            if progress_callback is not None
            else None
        ),
    )

    _report(f"Interpolating relative humidity to {target_xy.shape[0]} target cells...", rh_start, progress_callback)
    rh_interp = idw_interpolate(
        sample_xy,
        rh_values,
        target_xy,
        neighbors,
        power,
        chunk_size,
        chunk_callback=(
            (
                lambda ratio: progress_callback(
                    int(rh_start + ratio * (rh_end - rh_start)),
                    "Interpolating relative humidity...",
                )
            )
            if progress_callback is not None
            else None
        ),
    )

    wind_speed_arr = np.full(meta["height"] * meta["width"], NODATA, dtype="float32")
    wind_u_arr = np.full(meta["height"] * meta["width"], NODATA, dtype="float32")
    wind_v_arr = np.full(meta["height"] * meta["width"], NODATA, dtype="float32")
    rh_arr = np.full(meta["height"] * meta["width"], NODATA, dtype="float32")
    flat_idx = row_idx * meta["width"] + col_idx
    wind_speed_arr[flat_idx] = wind_speed_interp
    wind_u_arr[flat_idx] = wind_u_interp
    wind_v_arr[flat_idx] = wind_v_interp
    rh_arr[flat_idx] = rh_interp

    return (
        wind_speed_arr.reshape((meta["height"], meta["width"])),
        wind_u_arr.reshape((meta["height"], meta["width"])),
        wind_v_arr.reshape((meta["height"], meta["width"])),
        rh_arr.reshape((meta["height"], meta["width"])),
        meta,
    )


def write_raster(path: Path, arr: np.ndarray, meta: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    out_meta = meta.copy()
    out_meta.update(dtype="float32", count=1, nodata=float(NODATA), compress="lzw")
    with rasterio.open(path, "w", **out_meta) as dst:
        dst.write(arr.astype("float32"), 1)


def build_openmeteo_rasters(
    h3_resolution: int = 5,
    batch_size: int = 48,
    idw_neighbors: int = 8,
    idw_power: float = 2.0,
    interp_chunk_size: int = 20000,
    timeout_sec: int = 30,
    timezone: str = "UTC",
    wind_speed_unit: str = "ms",
    target_hour: str | None = None,
    u_out: Path | None = None,
    v_out: Path | None = None,
    rh_out: Path | None = None,
    request_retries: int = 4,
    progress_callback: ProgressCallback | None = None,
) -> tuple[Path, Path, Path]:
    target_hour = _normalize_target_hour(target_hour)
    u_path = Path(u_out) if u_out is not None else OUT_U_DIR / "openmeteo_wind_u_r5_now.tif"
    v_path = Path(v_out) if v_out is not None else OUT_V_DIR / "openmeteo_wind_v_r5_now.tif"
    rh_path = Path(rh_out) if rh_out is not None else OUT_RH_DIR / "openmeteo_rh_r5_now.tif"

    _report("Building weather domain from BP footprint...", 5, progress_callback)
    area_wgs84 = _valid_bp_geometry_wgs84(BP_PATH)
    _report("BP footprint geometry is ready.", 8, progress_callback)

    _report(f"Generating H3 sampling cells at resolution {h3_resolution}...", 10, progress_callback)
    cells = _h3_cells_from_geometry(area_wgs84, h3_resolution)
    if not cells:
        raise RuntimeError("No H3 cells generated for BP footprint.")
    _report(f"Generated {len(cells)} H3 cells.", 12, progress_callback)

    samples = fetch_openmeteo_samples(
        cells=cells,
        batch_size=batch_size,
        timeout_sec=timeout_sec,
        timezone=timezone,
        wind_speed_unit=wind_speed_unit,
        target_hour=target_hour,
        request_retries=request_retries,
        progress_callback=progress_callback,
        progress_start=15,
        progress_end=55,
    )

    wind_speed_arr, wind_u_arr, wind_v_arr, rh_arr, meta = interpolate_to_bp_grid(
        samples=samples,
        mask_geometry_wgs84=area_wgs84,
        neighbors=idw_neighbors,
        power=idw_power,
        chunk_size=interp_chunk_size,
        progress_callback=progress_callback,
        progress_start=55,
        progress_end=90,
    )

    _report(f"Writing wind U raster to {u_path} ...", 96, progress_callback)
    write_raster(u_path, wind_u_arr, meta)
    _report(f"Writing wind V raster to {v_path} ...", 96, progress_callback)
    write_raster(v_path, wind_v_arr, meta)
    _report(f"Writing RH raster to {rh_path} ...", 98, progress_callback)
    write_raster(rh_path, rh_arr, meta)
    _report("Done.", 100, progress_callback)
    return u_path, v_path, rh_path


def main() -> int:
    args = parse_args()
    build_openmeteo_rasters(
        h3_resolution=args.h3_resolution,
        batch_size=args.batch_size,
        idw_neighbors=args.idw_neighbors,
        idw_power=args.idw_power,
        interp_chunk_size=args.interp_chunk_size,
        timeout_sec=args.timeout_sec,
        timezone=args.timezone,
        wind_speed_unit=args.wind_speed_unit,
        target_hour=(None if args.use_current else (args.target_hour or None)),
        u_out=Path(args.u_out),
        v_out=Path(args.v_out),
        rh_out=Path(args.rh_out),
        request_retries=args.request_retries,
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise
