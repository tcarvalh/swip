from __future__ import annotations

import base64
import datetime as dt
import io
import json
import os
import threading
import uuid
from pathlib import Path

import numpy as np
import rasterio
from flask import Flask, jsonify, request, send_from_directory
from PIL import Image
from rasterio.warp import Resampling, calculate_default_transform, reproject
from werkzeug.utils import secure_filename

from build_openmeteo_rasters import build_openmeteo_rasters
from wipp_core import ModelParams, ModelRunInputs, load_uploaded_ignition_points, run_model

APP_DIR = Path(__file__).resolve().parent
BP_DIR = APP_DIR / "BP"
OPENMETEO_U_DIR = APP_DIR / "openmeteo_windu_rast"
OPENMETEO_V_DIR = APP_DIR / "openmeteo_windv_rast"
OPENMETEO_RH_DIR = APP_DIR / "openmeteo_rh_rast"
RUNTIME_DIR = Path(os.environ.get("LOCALAPPDATA", str(APP_DIR))) / "SWIP_Runtime"
MODEL_DEFAULTS_PATH = APP_DIR / "model_defaults.json"


def _load_model_defaults() -> ModelParams:
    defaults = ModelParams()
    if not MODEL_DEFAULTS_PATH.exists():
        return defaults

    with MODEL_DEFAULTS_PATH.open("r", encoding="utf-8") as f:
        payload = json.load(f)

    allowed_keys = {
        "bp_exponent",
        "wind_k",
        "spread_rate",
        "local_score_threshold",
        "cost_decay_lambda",
        "wind_support_min",
        "wind_support_max",
        "max_cumulative_cost",
        "rh_full_support",
        "rh_extinction_like",
        "moisture_support_floor",
    }
    values = {key: payload[key] for key in allowed_keys if key in payload}
    return ModelParams(**values)


APP_MODEL_DEFAULTS = _load_model_defaults()

UPLOADS_DIR = RUNTIME_DIR / "uploads"
OUTPUTS_DIR = RUNTIME_DIR / "outputs"
UPLOADS_DIR.mkdir(parents=True, exist_ok=True)
OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
OPENMETEO_U_DIR.mkdir(parents=True, exist_ok=True)
OPENMETEO_V_DIR.mkdir(parents=True, exist_ok=True)
OPENMETEO_RH_DIR.mkdir(parents=True, exist_ok=True)

RASTER_PREVIEW_CONFIG = {
    "bp": {
        "folder": BP_DIR,
        "label": "BP",
        "hide_zero": True,
        "opacity": 0.8,
        "palette": [(255, 245, 204), (255, 230, 153), (252, 141, 89), (227, 74, 51), (179, 0, 0)],
    },
    "om_v": {
        "folder": OPENMETEO_V_DIR,
        "label": "Open-Meteo wind V",
        "hide_zero": False,
        "opacity": 0.8,
        "palette": [(140, 81, 10), (216, 179, 101), (245, 245, 245), (90, 180, 172), (1, 102, 94)],
    },
    "om_u": {
        "folder": OPENMETEO_U_DIR,
        "label": "Open-Meteo wind U",
        "hide_zero": False,
        "opacity": 0.8,
        "palette": [(255, 245, 235), (254, 224, 210), (252, 187, 161), (252, 146, 114), (203, 24, 29)],
    },
    "om_rh": {
        "folder": OPENMETEO_RH_DIR,
        "label": "Open-Meteo RH",
        "hide_zero": False,
        "opacity": 0.8,
        "palette": [(255, 255, 229), (199, 233, 180), (127, 205, 187), (65, 182, 196), (34, 94, 168)],
    },
}

app = Flask(__name__, static_folder=".", static_url_path="")

OPENMETEO_U_DEFAULT = "openmeteo_wind_u_r5_now.tif"
OPENMETEO_V_DEFAULT = "openmeteo_wind_v_r5_now.tif"
OPENMETEO_RH_DEFAULT = "openmeteo_rh_r5_now.tif"

_OPENMETEO_LOCK = threading.Lock()
_OPENMETEO_THREAD: threading.Thread | None = None
_OPENMETEO_STATE: dict = {
    "state": "idle",
    "progress": 0,
    "message": "Not started.",
    "error": "",
    "used_cache": False,
    "u_file": "",
    "v_file": "",
    "rh_file": "",
    "target_hour": None,
    "started_at": None,
    "updated_at": None,
    "finished_at": None,
}


def _utc_now_iso() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def _normalize_target_hour(raw: str | None) -> str | None:
    value = (raw or "").strip()
    if not value:
        return None
    try:
        parsed = dt.datetime.strptime(value, "%Y-%m-%dT%H:%M")
    except ValueError as exc:
        raise ValueError("weather time must use YYYY-MM-DDTHH:00 format.") from exc
    if parsed.minute != 0:
        raise ValueError("weather time must be specified to the hour.")
    return parsed.strftime("%Y-%m-%dT%H:%M")


def _openmeteo_cached_files() -> tuple[str, str, str]:
    u_files = list_tif_files(OPENMETEO_U_DIR)
    v_files = list_tif_files(OPENMETEO_V_DIR)
    rh_files = list_tif_files(OPENMETEO_RH_DIR)
    u_file = u_files[0] if u_files else ""
    v_file = v_files[0] if v_files else ""
    rh_file = rh_files[0] if rh_files else ""
    return u_file, v_file, rh_file


def _openmeteo_has_cache() -> bool:
    u_file, v_file, rh_file = _openmeteo_cached_files()
    return bool(u_file and v_file and rh_file)


def _openmeteo_snapshot_locked() -> dict:
    return dict(_OPENMETEO_STATE)


def _openmeteo_snapshot() -> dict:
    with _OPENMETEO_LOCK:
        return _openmeteo_snapshot_locked()


def _set_openmeteo_state_locked(**kwargs) -> None:
    _OPENMETEO_STATE.update(kwargs)
    _OPENMETEO_STATE["updated_at"] = _utc_now_iso()


def _progress_update(progress: int, message: str) -> None:
    with _OPENMETEO_LOCK:
        _set_openmeteo_state_locked(
            state="running",
            progress=max(0, min(100, int(progress))),
            message=message,
            error="",
            used_cache=False,
        )


def _run_openmeteo_build_job(target_hour: str | None) -> None:
    global _OPENMETEO_THREAD
    try:
        u_out = OPENMETEO_U_DIR / OPENMETEO_U_DEFAULT
        v_out = OPENMETEO_V_DIR / OPENMETEO_V_DEFAULT
        rh_out = OPENMETEO_RH_DIR / OPENMETEO_RH_DEFAULT
        build_openmeteo_rasters(
            u_out=u_out,
            v_out=v_out,
            rh_out=rh_out,
            target_hour=target_hour,
            progress_callback=_progress_update,
        )
        u_file, v_file, rh_file = _openmeteo_cached_files()
        with _OPENMETEO_LOCK:
            _set_openmeteo_state_locked(
                state="ready",
                progress=100,
                message=(f"Open-Meteo rasters are ready for {target_hour}." if target_hour else "Open-Meteo rasters are ready for current weather."),
                error="",
                used_cache=False,
                u_file=u_file,
                v_file=v_file,
                rh_file=rh_file,
                target_hour=target_hour,
                finished_at=_utc_now_iso(),
            )
    except Exception as exc:
        with _OPENMETEO_LOCK:
            _set_openmeteo_state_locked(
                state="error",
                message="Failed to build Open-Meteo rasters.",
                error=str(exc),
                finished_at=_utc_now_iso(),
            )
    finally:
        with _OPENMETEO_LOCK:
            _OPENMETEO_THREAD = None


def _ensure_openmeteo_rasters(force: bool, target_hour: str | None) -> dict:
    global _OPENMETEO_THREAD
    with _OPENMETEO_LOCK:
        if _OPENMETEO_THREAD is not None and _OPENMETEO_THREAD.is_alive():
            return _openmeteo_snapshot_locked()

        requested_hour = _normalize_target_hour(target_hour)
        u_file, v_file, rh_file = _openmeteo_cached_files()
        if not force and u_file and v_file and rh_file and _OPENMETEO_STATE.get("target_hour") == requested_hour:
            _set_openmeteo_state_locked(
                state="ready",
                progress=100,
                message=(f"Using cached Open-Meteo rasters for {requested_hour}." if requested_hour else "Using cached Open-Meteo rasters for current weather."),
                error="",
                used_cache=True,
                u_file=u_file,
                v_file=v_file,
                rh_file=rh_file,
                target_hour=requested_hour,
                finished_at=_utc_now_iso(),
            )
            return _openmeteo_snapshot_locked()

        _set_openmeteo_state_locked(
            state="running",
            progress=1,
            message=(f"Starting Open-Meteo raster build for {requested_hour}..." if requested_hour else "Starting Open-Meteo raster build for current weather..."),
            error="",
            used_cache=False,
            u_file="",
            v_file="",
            rh_file="",
            target_hour=requested_hour,
            started_at=_utc_now_iso(),
            finished_at=None,
        )
        _OPENMETEO_THREAD = threading.Thread(target=_run_openmeteo_build_job, args=(requested_hour,), daemon=True)
        _OPENMETEO_THREAD.start()
        return _openmeteo_snapshot_locked()


@app.after_request
def add_cors_headers(response):
    response.headers["Access-Control-Allow-Origin"] = "*"
    response.headers["Access-Control-Allow-Methods"] = "GET, POST, OPTIONS"
    response.headers["Access-Control-Allow-Headers"] = "Content-Type"
    return response


def list_tif_files(folder: Path) -> list[str]:
    if not folder.exists():
        return []
    files = [p.name for p in folder.iterdir() if p.is_file() and p.suffix.lower() in {".tif", ".tiff"}]
    files.sort()
    return files


def resolve_local_raster(folder: Path, file_name: str, label: str) -> Path:
    clean_name = (file_name or "").strip()
    if not clean_name:
        raise ValueError(f"Missing required input: {label}.")
    if Path(clean_name).name != clean_name:
        raise ValueError(f"Invalid {label} file name.")

    full_path = folder / clean_name
    if not full_path.exists() or full_path.suffix.lower() not in {".tif", ".tiff"}:
        raise ValueError(f"Selected {label} file was not found in {folder.name}.")
    return full_path


def _jenks_breaks(values: np.ndarray, n_classes: int) -> list[float]:
    arr = np.asarray(values, dtype="float64")
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        raise ValueError("No finite values available for Jenks classification.")

    arr.sort()
    unique = np.unique(arr)
    if unique.size == 1:
        return [float(unique[0])] * (n_classes + 1)

    if arr.size > 10000:
        idx = np.linspace(0, arr.size - 1, 10000, dtype=int)
        arr = arr[idx]
        arr.sort()
        unique = np.unique(arr)

    n = arr.size
    k = max(1, min(n_classes, unique.size))
    lower = np.zeros((n + 1, k + 1), dtype=int)
    var = np.full((n + 1, k + 1), np.inf, dtype="float64")

    for i in range(1, k + 1):
        lower[1, i] = 1
        var[1, i] = 0.0

    for l in range(2, n + 1):
        s1 = 0.0
        s2 = 0.0
        w = 0.0
        for m in range(1, l + 1):
            idx = l - m
            val = arr[idx]
            s1 += val
            s2 += val * val
            w += 1.0
            variance = s2 - (s1 * s1) / w
            if idx != 0:
                for j in range(2, k + 1):
                    candidate = variance + var[idx, j - 1]
                    if candidate < var[l, j]:
                        lower[l, j] = idx + 1
                        var[l, j] = candidate
        lower[l, 1] = 1
        var[l, 1] = variance

    breaks = [0.0] * (k + 1)
    breaks[k] = float(arr[-1])
    count = n
    for j in range(k, 1, -1):
        idx = lower[count, j] - 1
        breaks[j - 1] = float(arr[idx])
        count = lower[count, j] - 1
    breaks[0] = float(arr[0])

    while len(breaks) < n_classes + 1:
        breaks.append(breaks[-1])
    return breaks


def _apply_jenks_palette(raster: np.ndarray, valid: np.ndarray, palette: list[tuple[int, int, int]]) -> np.ndarray:
    values = raster[valid].astype("float64")
    breaks = _jenks_breaks(values, len(palette))
    rgb = np.zeros(raster.shape + (3,), dtype=np.uint8)

    for idx, color in enumerate(palette):
        lower = breaks[idx]
        upper = breaks[idx + 1]
        if idx == len(palette) - 1:
            cls_mask = valid & (raster >= lower) & (raster <= upper)
        else:
            cls_mask = valid & (raster >= lower) & (raster < upper)
        rgb[cls_mask] = np.array(color, dtype=np.uint8)

    unassigned = valid & np.all(rgb == 0, axis=2)
    if np.any(unassigned):
        rgb[unassigned] = np.array(palette[-1], dtype=np.uint8)
    return rgb


def render_raster_preview(path: Path, layer_key: str) -> tuple[str, list[list[float]]]:
    cfg = RASTER_PREVIEW_CONFIG[layer_key]
    with rasterio.open(path) as src:
        src_data = src.read(1).astype("float32")
        src_nodata = src.nodata
        if src.crs is None:
            raise ValueError(f"{cfg['label']} raster has no CRS.")

        if str(src.crs).upper() != "EPSG:4326":
            dst_transform, dst_w, dst_h = calculate_default_transform(
                src.crs, "EPSG:4326", src.width, src.height, *src.bounds
            )
            raster = np.full((dst_h, dst_w), np.nan, dtype="float32")
            reproject(
                source=src_data,
                destination=raster,
                src_transform=src.transform,
                src_crs=src.crs,
                src_nodata=src_nodata,
                dst_transform=dst_transform,
                dst_crs="EPSG:4326",
                dst_nodata=np.nan,
                resampling=Resampling.nearest,
            )
            affine = dst_transform
            width = dst_w
            height = dst_h
            nodata = np.nan
        else:
            raster = src_data
            affine = src.transform
            width = src.width
            height = src.height
            nodata = src_nodata

        mask = ~np.isfinite(raster)
        if nodata is not None:
            mask |= raster == nodata
        if cfg["hide_zero"]:
            mask |= raster <= 0.0

        valid = ~mask
        if not np.any(valid):
            raise ValueError(f"{cfg['label']} raster has no valid cells to visualize.")

        rgb = _apply_jenks_palette(raster, valid, cfg["palette"])
        alpha = np.where(valid, int(float(cfg["opacity"]) * 255), 0).astype(np.uint8)
        rgba = np.dstack((rgb, alpha))

        img = Image.fromarray(rgba, mode="RGBA")
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        encoded = base64.b64encode(buf.getvalue()).decode("ascii")

        left = float(affine.c)
        top = float(affine.f)
        right = float(affine.c + affine.a * width)
        bottom = float(affine.f + affine.e * height)
        min_lon = min(left, right)
        max_lon = max(left, right)
        min_lat = min(bottom, top)
        max_lat = max(bottom, top)
        bounds = [[min_lat, min_lon], [max_lat, max_lon]]

    return f"data:image/png;base64,{encoded}", bounds


@app.get("/")
def index():
    return send_from_directory(APP_DIR, "index.html")


@app.get("/bp-files")
def bp_files_endpoint():
    return jsonify({"bp_files": list_tif_files(BP_DIR)})


@app.get("/openmeteo-v-files")
def openmeteo_v_files_endpoint():
    return jsonify({"openmeteo_v_files": list_tif_files(OPENMETEO_V_DIR)})


@app.get("/openmeteo-u-files")
def openmeteo_u_files_endpoint():
    return jsonify({"openmeteo_u_files": list_tif_files(OPENMETEO_U_DIR)})


@app.get("/openmeteo-rh-files")
def openmeteo_rh_files_endpoint():
    return jsonify({"openmeteo_rh_files": list_tif_files(OPENMETEO_RH_DIR)})


@app.get("/openmeteo/build-status")
def openmeteo_build_status_endpoint():
    with _OPENMETEO_LOCK:
        if _OPENMETEO_STATE.get("state") == "idle":
            u_file, v_file, rh_file = _openmeteo_cached_files()
            if u_file and v_file and rh_file:
                _set_openmeteo_state_locked(
                    state="ready",
                    progress=100,
                    message="Using cached Open-Meteo rasters.",
                    error="",
                    used_cache=True,
                    u_file=u_file,
                    v_file=v_file,
                    rh_file=rh_file,
                    finished_at=_utc_now_iso(),
                )
        status = _openmeteo_snapshot_locked()
    status["cache_available"] = _openmeteo_has_cache()
    return jsonify(status)


@app.post("/openmeteo/build")
def openmeteo_build_endpoint():
    payload = request.get_json(silent=True) or {}
    force = bool(payload.get("force", False))
    use_current = bool(payload.get("use_current", False))
    target_hour = None if use_current else _normalize_target_hour(payload.get("target_hour"))
    status = _ensure_openmeteo_rasters(force=force, target_hour=target_hour)
    status["cache_available"] = _openmeteo_has_cache()
    return jsonify(status)


@app.get("/raster-preview")
def raster_preview_endpoint():
    try:
        layer = (request.args.get("layer", "") or "").strip().lower()
        file_name = request.args.get("file_name", "")

        if layer not in RASTER_PREVIEW_CONFIG:
            raise ValueError("Invalid layer. Use one of: bp, om_v, om_u, om_rh.")

        cfg = RASTER_PREVIEW_CONFIG[layer]
        raster_path = resolve_local_raster(cfg["folder"], file_name, cfg["label"])
        image_data_url, bounds = render_raster_preview(raster_path, layer)

        return jsonify(
            {
                "layer": layer,
                "file_name": raster_path.name,
                "image_data_url": image_data_url,
                "bounds": bounds,
            }
        )
    except Exception as exc:
        return jsonify({"error": str(exc)}), 400


@app.post("/ignition-preview")
def ignition_preview_endpoint():
    try:
        ignition_file = request.files.get("ignition_file")
        if ignition_file is None or not ignition_file.filename:
            raise ValueError("Upload an ignition vector file to preview it on the map.")

        preview_id = "preview_" + uuid.uuid4().hex[:8]
        preview_dir = UPLOADS_DIR / preview_id
        preview_dir.mkdir(parents=True, exist_ok=True)
        ignition_path = preview_dir / secure_filename(ignition_file.filename)
        ignition_file.save(ignition_path)

        ign_gdf = load_uploaded_ignition_points(ignition_path, "EPSG:4326")
        if ign_gdf.empty:
            raise ValueError("Ignition file did not contain any features.")

        geom = ign_gdf.geometry.iloc[0]
        if geom is None or geom.is_empty:
            raise ValueError("Ignition file geometry is empty.")
        point = geom if geom.geom_type == "Point" else geom.representative_point()

        return jsonify(
            {
                "lat": float(point.y),
                "lng": float(point.x),
            }
        )
    except Exception as exc:
        return jsonify({"error": str(exc)}), 400


@app.post("/run")
def run_endpoint():
    try:
        bp_name = request.form.get("bp_name", "")
        u_name = request.form.get("u_name", "")
        v_name = request.form.get("v_name", "")
        ignition_file = request.files.get("ignition_file")

        ign_lat_raw = (request.form.get("ignition_lat") or "").strip()
        ign_lng_raw = (request.form.get("ignition_lng") or "").strip()
        ign_lat = float(ign_lat_raw) if ign_lat_raw else None
        ign_lng = float(ign_lng_raw) if ign_lng_raw else None
        exclude_zero = request.form.get("exclude_zero", "true").lower() == "true"
        rh_full_support = APP_MODEL_DEFAULTS.rh_full_support
        rh_extinction_like = APP_MODEL_DEFAULTS.rh_extinction_like
        if rh_extinction_like <= rh_full_support:
            raise ValueError("Configured RH Extinction-Like must be greater than RH Full Support.")
        bp_path = resolve_local_raster(BP_DIR, bp_name, "BP")

        if not u_name:
            om_u_files = list_tif_files(OPENMETEO_U_DIR)
            if not om_u_files:
                raise ValueError("Open-Meteo wind U raster is not available yet.")
            u_name = om_u_files[0]
        if not v_name:
            om_v_files = list_tif_files(OPENMETEO_V_DIR)
            if not om_v_files:
                raise ValueError("Open-Meteo wind V raster is not available yet.")
            v_name = om_v_files[0]
        om_rh_files = list_tif_files(OPENMETEO_RH_DIR)
        rh_name = om_rh_files[0] if om_rh_files else ""

        # Simulation now uses Open-Meteo-generated U/V vector rasters plus optional RH.
        wind_u_path = resolve_local_raster(OPENMETEO_U_DIR, u_name, "Open-Meteo wind U")
        wind_v_path = resolve_local_raster(OPENMETEO_V_DIR, v_name, "Open-Meteo wind V")
        rh_path = resolve_local_raster(OPENMETEO_RH_DIR, rh_name, "Open-Meteo RH") if rh_name else None

        run_id = dt.datetime.now().strftime("%Y%m%d_%H%M%S") + "_" + uuid.uuid4().hex[:8]
        run_upload_dir = UPLOADS_DIR / run_id
        run_output_dir = OUTPUTS_DIR / run_id
        run_upload_dir.mkdir(parents=True, exist_ok=True)
        run_output_dir.mkdir(parents=True, exist_ok=True)

        uploaded_ignition_gdf = None
        if ignition_file is not None and ignition_file.filename:
            ignition_path = run_upload_dir / secure_filename(ignition_file.filename)
            ignition_file.save(ignition_path)
            uploaded_ignition_gdf = load_uploaded_ignition_points(ignition_path, "EPSG:4326")

        if uploaded_ignition_gdf is None and (ign_lat is None or ign_lng is None):
            raise ValueError("Set an ignition point on the map or upload an ignition vector file.")

        result = run_model(
            ModelRunInputs(
                bp_path=bp_path,
                wind_u_path=wind_u_path,
                wind_v_path=wind_v_path,
                rh_path=rh_path,
                ign_lat=ign_lat,
                ign_lng=ign_lng,
                ign_gdf=uploaded_ignition_gdf,
                exclude_zero=exclude_zero,
                run_output_dir=run_output_dir,
                params=ModelParams(
                    bp_exponent=APP_MODEL_DEFAULTS.bp_exponent,
                    wind_k=APP_MODEL_DEFAULTS.wind_k,
                    spread_rate=APP_MODEL_DEFAULTS.spread_rate,
                    local_score_threshold=APP_MODEL_DEFAULTS.local_score_threshold,
                    cost_decay_lambda=APP_MODEL_DEFAULTS.cost_decay_lambda,
                    wind_support_min=APP_MODEL_DEFAULTS.wind_support_min,
                    wind_support_max=APP_MODEL_DEFAULTS.wind_support_max,
                    max_cumulative_cost=APP_MODEL_DEFAULTS.max_cumulative_cost,
                    rh_full_support=rh_full_support,
                    rh_extinction_like=rh_extinction_like,
                    moisture_support_floor=APP_MODEL_DEFAULTS.moisture_support_floor,
                ),
            )
        )

        return jsonify(
            {
                "summary": result.summary,
                "total_area_acres": result.total_area_acres,
                "geojson": result.geojson,
            }
        )

    except Exception as exc:
        return jsonify({"error": str(exc)}), 400


if __name__ == "__main__":
    debug_enabled = os.environ.get("SWIP_DEBUG", "").strip().lower() in {"1", "true", "yes"}
    app.run(
        host="127.0.0.1",
        port=5000,
        debug=debug_enabled,
        threaded=True,
        use_reloader=debug_enabled,
    )
