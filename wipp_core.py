"""Core wildfire spread logic for WIPP 2.0.

This module contains the raster-to-raster spread model that powers the app.
Conceptually, it combines:

1. A static burn-probability raster that acts as a background susceptibility
   surface.
2. Event-specific wind rasters that bias spread direction and relative ease of
   movement from cell to cell.
3. A relative-humidity raster used here as a provisional moisture proxy that
   dampens spread support in wetter areas.
4. One or more ignition points that seed the spread simulation.

Scientifically, this is best understood as a pragmatic spread-potential model,
not a full physical fire-behavior simulator. The algorithm uses a
cost-distance-style traversal with local gating rules to estimate which cells
remain reachable under the combined BP, wind, and RH controls.
"""

from __future__ import annotations

import csv
import heapq
import json
import math
import os
import zipfile
from dataclasses import dataclass, field
from pathlib import Path

import geopandas as gpd
import h3.api.basic_int as h3
import numpy as np
import rasterio
from rasterio.features import shapes
from rasterio.transform import rowcol, xy
from rasterio.warp import Resampling, reproject, transform as warp_transform
from shapely.geometry import Point, shape
from shapely.ops import unary_union


@dataclass(slots=True)
class ModelParams:
    """User-tunable controls for the spread surrogate model.

    These parameters do not represent a single published fire model directly.
    Instead, they expose the main controls on how easily spread propagates
    through the raster graph:

    - `bp_exponent` changes how strongly the burn-probability raster influences
      spread support and travel cost.
    - `wind_k` controls how strongly wind alignment affects spread.
    - `spread_rate` globally speeds up or slows down edge traversal.
    - `local_score_threshold` is the core stopping rule: if a candidate next
      cell does not meet this local spread-support threshold, the path stops.
    - `cost_decay_lambda` reduces support as cumulative travel cost grows,
      which suppresses very long or weakly supported spread paths.
    - RH-related parameters convert RH into a simple moisture-support surface.
    """
    bp_exponent: float = 1.0
    wind_k: float = 0.5
    spread_rate: float = 1.0
    local_score_threshold: float = 0.15
    cost_decay_lambda: float = 0.01
    wind_support_min: float = 0.25
    wind_support_max: float = 1.75
    max_cumulative_cost: float | None = None
    # These RH parameters are provisional surrogate controls, not a literal
    # fuel-moisture extinction model.
    rh_full_support: float = 15.0
    rh_extinction_like: float = 70.0
    moisture_support_floor: float = 0.05


@dataclass(slots=True)
class ModelRunResult:
    """Packaged outputs returned to the app after one model run."""
    summary: list[dict]
    total_area_acres: float
    geojson: dict
    burnmask_path: Path
    summary_csv_path: Path
    burned_cells: int
    params: ModelParams


@dataclass(slots=True)
class ModelRunInputs:
    """All external inputs required to execute one model run."""
    bp_path: Path
    wind_u_path: Path
    wind_v_path: Path
    rh_path: Path  # required RH raster used as a provisional moisture proxy
    ign_lat: float | None = None
    ign_lng: float | None = None
    ign_gdf: gpd.GeoDataFrame | None = None
    exclude_zero: bool = True
    run_output_dir: Path | None = None
    params: ModelParams = field(default_factory=ModelParams)


def load_uploaded_ignition_points(path: Path, target_crs: str) -> gpd.GeoDataFrame:
    """Load user-supplied ignition features and project them to the model grid.

    The app allows either direct spatial files or zipped shapefiles. The model
    ultimately needs ignition features in the same CRS as the BP raster so they
    can be converted to raster seed cells reliably.
    """
    if path.suffix.lower() == ".zip":
        extract_dir = path.with_suffix("")
        extract_dir.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(path, "r") as zf:
            shp_names = [name for name in zf.namelist() if name.lower().endswith(".shp")]
            if not shp_names:
                raise ValueError("Ignition ZIP does not contain a .shp file.")
            zf.extractall(extract_dir)

        shp_path = extract_dir / shp_names[0]
        # Some uploaded shapefile ZIPs omit or corrupt the .shx index. GDAL can
        # rebuild it on read when this flag is enabled.
        os.environ.setdefault("SHAPE_RESTORE_SHX", "YES")
        ign_gdf = gpd.read_file(shp_path)
    else:
        ign_gdf = gpd.read_file(path)

    if ign_gdf.empty:
        raise ValueError("Ignition upload contains no features.")
    if ign_gdf.crs is None:
        raise ValueError("Ignition upload is missing a CRS.")
    return ign_gdf.to_crs(target_crs)


def warp_to_grid(src_path: Path, grid_meta: dict, resampling: Resampling) -> np.ndarray:
    """Resample a raster input onto the BP raster grid.

    The BP raster defines the working analysis grid. Wind and RH are therefore
    reprojected/resampled onto that same extent, resolution, transform, and CRS
    so all per-cell calculations are aligned.
    """
    with rasterio.open(src_path) as src:
        dst = np.empty((grid_meta["height"], grid_meta["width"]), dtype="float32")
        reproject(
            source=rasterio.band(src, 1),
            destination=dst,
            src_transform=src.transform,
            src_crs=src.crs,
            dst_transform=grid_meta["transform"],
            dst_crs=grid_meta["crs"],
            dst_nodata=np.nan,
            resampling=resampling,
        )
    return dst


def build_bp_surfaces(
    burn_raster: np.ndarray,
    valid_burnable: np.ndarray,
    params: ModelParams,
    epsilon: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Derive BP-driven support and travel-cost surfaces.

    BP acts here as a static susceptibility prior. Higher BP means the model
    treats that cell as more supportive of spread. We use it two ways:

    - `bp_support`: a normalized 0-1 style support surface used in the local
      threshold check.
    - `base_cost`: an inverse-cost surface used in the Dijkstra-style traversal,
      where more supportive cells are cheaper to enter.
    """
    bp_term = np.power(np.maximum(burn_raster, epsilon), float(params.bp_exponent)).astype("float32")
    bp_term = np.where(valid_burnable, bp_term, 0.0).astype("float32")

    if np.any(valid_burnable):
        bp_term_max = float(np.max(bp_term[valid_burnable]))
    else:
        bp_term_max = 0.0

    if bp_term_max > 0.0 and math.isfinite(bp_term_max):
        bp_support = np.where(valid_burnable, bp_term / bp_term_max, 0.0).astype("float32")
    else:
        bp_support = np.zeros_like(bp_term, dtype="float32")

    base_cost = (1.0 / np.maximum(bp_term, epsilon)).astype("float32")
    base_cost = np.where(valid_burnable, base_cost, 1_000_000.0).astype("float32")
    return bp_term, bp_support, base_cost


def build_moisture_support_from_rh(
    rh_grid: np.ndarray | None,
    params: ModelParams,
    shape_: tuple[int, int],
) -> np.ndarray:
    """Map RH to a simple moisture-support surface.

    RH is used here as a stand-in proxy for moisture effects. This is
    intentionally simple and meant to be replaced later by a more physically
    grounded fuel-moisture proxy.

    Interpretation:
    - low RH => drier fuels => stronger support for spread
    - high RH => wetter conditions => weaker support for spread

    The output is a bounded support surface rather than a literal moisture or
    extinction calculation.
    """
    floor_ = min(1.0, max(0.0, float(params.moisture_support_floor)))
    if float(params.rh_extinction_like) <= float(params.rh_full_support):
        raise ValueError("rh_extinction_like must be greater than rh_full_support.")

    if rh_grid is None:
        return np.ones(shape_, dtype="float32")

    span = max(float(params.rh_extinction_like) - float(params.rh_full_support), 1e-6)
    rh_clamped = np.clip(rh_grid.astype("float32"), 0.0, 100.0)
    rh_safe = np.where(np.isfinite(rh_clamped), rh_clamped, float(params.rh_full_support))

    dryness = (float(params.rh_extinction_like) - rh_safe) / span
    dryness = np.clip(dryness, 0.0, 1.0)

    moisture_support = floor_ + (1.0 - floor_) * dryness
    moisture_support = np.clip(moisture_support, floor_, 1.0).astype("float32")
    return moisture_support



def compute_wind_support(
    wind_u: np.ndarray,
    wind_v: np.ndarray,
    r: int,
    c: int,
    rr: int,
    cc: int,
    movement_x: float,
    movement_y: float,
    wind_speed_ref: float,
    params: ModelParams,
) -> tuple[float, float]:
    """Estimate how wind changes local spread support and travel cost.

    The model compares the candidate movement direction to the local wind
    vector. Movement aligned with wind receives higher support; movement
    against wind is penalized. Two related outputs are returned:

    - `wind_support`: used in the local spread-score gate
    - `travel_wind_factor`: applied to edge cost so aligned movement is cheaper
    """
    # Treat missing wind as neutral forcing instead of blocking spread.
    u0 = wind_u[r, c] if np.isfinite(wind_u[r, c]) else 0.0
    u1 = wind_u[rr, cc] if np.isfinite(wind_u[rr, cc]) else 0.0
    v0 = wind_v[r, c] if np.isfinite(wind_v[r, c]) else 0.0
    v1 = wind_v[rr, cc] if np.isfinite(wind_v[rr, cc]) else 0.0

    local_u = float((u0 + u1) / 2.0)
    local_v = float((v0 + v1) / 2.0)
    wind_mag = math.hypot(local_u, local_v)
    if wind_mag > 0.0 and math.isfinite(wind_mag):
        alignment = (movement_x * (local_u / wind_mag)) + (movement_y * (local_v / wind_mag))
    else:
        alignment = 0.0

    speed_scale = min(1.0, wind_mag / wind_speed_ref) if wind_speed_ref > 0.0 else 0.0
    wind_support = 1.0 + (float(params.wind_k) * alignment * speed_scale)
    wind_support = max(float(params.wind_support_min), min(float(params.wind_support_max), wind_support))

    # Preserve the current directional wind effect on travel cost while allowing
    # spread_rate to globally ease propagation through lower effective edge cost.
    travel_wind_factor = max(0.25, min(2.0, 1.0 - (float(params.wind_k) * alignment)))
    return wind_support, travel_wind_factor


def compute_local_spread_score(
    bp_support_value: float,
    wind_support: float,
    moisture_value: float,
    cumulative_cost: float,
    params: ModelParams,
) -> float:
    """Combine local controls into a spread-sustainability score.

    This is not a literal ignition probability. It is a heuristic local score
    that answers: "Given BP support, wind support, moisture support, and the
    cumulative cost paid so far, is continued spread into this neighbor still
    plausible enough to keep exploring?"
    """
    cost_decay = math.exp(-float(params.cost_decay_lambda) * cumulative_cost) if float(params.cost_decay_lambda) > 0 else 1.0
    # local_score is a spread-sustainability surrogate score, not a literal ignition probability.
    return float(bp_support_value) * float(wind_support) * float(moisture_value) * cost_decay


def summarize_quantiles(values: list[float], prefix: str) -> dict[str, float | int | str]:
    """Return simple diagnostics for model transparency and tuning."""
    if not values:
        return {
            f"{prefix}_count": 0,
            f"{prefix}_q00": "",
            f"{prefix}_q25": "",
            f"{prefix}_q50": "",
            f"{prefix}_q75": "",
            f"{prefix}_q100": "",
        }

    arr = np.asarray(values, dtype="float64")
    return {
        f"{prefix}_count": int(arr.size),
        f"{prefix}_q00": float(np.quantile(arr, 0.00)),
        f"{prefix}_q25": float(np.quantile(arr, 0.25)),
        f"{prefix}_q50": float(np.quantile(arr, 0.50)),
        f"{prefix}_q75": float(np.quantile(arr, 0.75)),
        f"{prefix}_q100": float(np.quantile(arr, 1.00)),
    }


def dijkstra_local_score_reach(
    base_cost: np.ndarray,
    bp_support: np.ndarray,
    wind_u: np.ndarray,
    wind_v: np.ndarray,
    start_rcs: list[tuple[int, int]],
    blocked: np.ndarray | None = None,
    use_8_neighbor: bool = True,
    params: ModelParams | None = None,
    moisture_support: np.ndarray | None = None,
    diagnostics: dict | None = None,
) -> np.ndarray:
    """Run the core graph traversal over the raster grid.

    This is the heart of the model. It behaves like a Dijkstra-style search on
    the raster cell graph:

    - raster cells are nodes
    - neighbor moves are edges
    - BP and wind shape the travel cost
    - a local score threshold decides whether spread can continue into a cell

    The result is a binary burn mask of reachable cells rather than a time-
    explicit fire simulation.
    """
    active_params = params or ModelParams()
    h, w = base_cost.shape
    if blocked is None:
        blocked = np.zeros((h, w), dtype=bool)
    if moisture_support is None:
        moisture_support = np.ones((h, w), dtype="float32")

    for (r, c) in start_rcs:
        if 0 <= r < h and 0 <= c < w:
            blocked[r, c] = False

    wind_mag_grid = np.sqrt(
        np.square(np.where(np.isfinite(wind_u), wind_u, 0.0).astype("float32"))
        + np.square(np.where(np.isfinite(wind_v), wind_v, 0.0).astype("float32"))
    )
    finite_wind = np.isfinite(wind_mag_grid)
    wind_speed_ref = float(np.max(wind_mag_grid[finite_wind])) if np.any(finite_wind) else 0.0
    if not math.isfinite(wind_speed_ref) or wind_speed_ref < 0.0:
        wind_speed_ref = 0.0

    dist = np.full((h, w), np.inf, dtype="float64")
    pq: list[tuple[float, int, int]] = []

    for (r, c) in start_rcs:
        if 0 <= r < h and 0 <= c < w and not blocked[r, c]:
            dist[r, c] = 0.0
            heapq.heappush(pq, (0.0, r, c))

    if not pq:
        return np.zeros((h, w), dtype=np.uint8)

    moves4 = [(-1, 0, 1.0), (1, 0, 1.0), (0, -1, 1.0), (0, 1, 1.0)]
    sqrt2 = math.sqrt(2.0)
    moves8 = moves4 + [(-1, -1, sqrt2), (-1, 1, sqrt2), (1, -1, sqrt2), (1, 1, sqrt2)]
    moves = moves8 if use_8_neighbor else moves4

    reached = np.zeros((h, w), dtype=np.uint8)
    epsilon = 1e-6
    accepted_local_scores: list[float] = []
    rejected_local_scores: list[float] = []
    cumulative_costs: list[float] = []

    # Pop the cheapest currently reachable cell, then test each neighbor for
    # whether spread can continue into it.
    while pq:
        d, r, c = heapq.heappop(pq)
        if d != dist[r, c]:
            continue

        reached[r, c] = 1

        for dr, dc, step in moves:
            rr = r + dr
            cc = c + dc
            if rr < 0 or rr >= h or cc < 0 or cc >= w:
                continue
            if blocked[rr, cc]:
                continue

            movement_x = dc / step
            movement_y = -dr / step
            wind_support, travel_wind_factor = compute_wind_support(
                wind_u,
                wind_v,
                r,
                c,
                rr,
                cc,
                movement_x,
                movement_y,
                wind_speed_ref,
                active_params,
            )

            c_cost = float(base_cost[r, c])
            n_cost = float(base_cost[rr, cc])
            edge_cost = step * ((c_cost + n_cost) / 2.0) * travel_wind_factor
            edge_cost /= max(float(active_params.spread_rate), epsilon)
            nd = d + edge_cost

            moisture_value = 1.0
            if moisture_support is not None:
                candidate_moisture = float(moisture_support[rr, cc])
                moisture_value = candidate_moisture if math.isfinite(candidate_moisture) else 1.0

            local_score = compute_local_spread_score(
                bp_support_value=float(bp_support[rr, cc]),
                wind_support=wind_support,
                moisture_value=moisture_value,
                cumulative_cost=nd,
                params=active_params,
            )

            # The main stopping rule: even if a path is geometrically possible,
            # it is rejected when local spread support falls below the threshold.
            if local_score < float(active_params.local_score_threshold):
                rejected_local_scores.append(local_score)
                continue
            if active_params.max_cumulative_cost is not None and nd > float(active_params.max_cumulative_cost):
                rejected_local_scores.append(local_score)
                continue
            if nd < dist[rr, cc]:
                accepted_local_scores.append(local_score)
                cumulative_costs.append(nd)
                dist[rr, cc] = nd
                heapq.heappush(pq, (nd, rr, cc))

    if diagnostics is not None:
        diagnostics.update(summarize_quantiles(accepted_local_scores, "accepted_local_score"))
        diagnostics.update(summarize_quantiles(rejected_local_scores, "rejected_local_score"))
        diagnostics.update(summarize_quantiles(cumulative_costs, "cumulative_cost"))

    return reached


def points_to_seed_cells(points_gdf: gpd.GeoDataFrame, transform, shape_: tuple[int, int]) -> list[tuple[int, int]]:
    """Convert ignition features into raster-cell seeds.

    The model ultimately ignites cells, not vector features. Point ignitions
    map directly, while non-point uploads use a representative interior point
    so each feature seeds a stable in-grid location.
    """
    seeds: list[tuple[int, int]] = []
    for geom in points_gdf.geometry:
        if geom is None or geom.is_empty:
            continue

        # Use representative points for non-point geometries so polygon and line
        # ignition uploads still map to a stable seed cell inside the feature.
        pt = geom if geom.geom_type == "Point" else geom.representative_point()
        r, c = rowcol(transform, pt.x, pt.y)
        if 0 <= r < shape_[0] and 0 <= c < shape_[1]:
            seeds.append((int(r), int(c)))

    return list(dict.fromkeys(seeds))


def write_u8_raster(path: Path, arr_u8: np.ndarray, meta: dict) -> None:
    """Write the binary burn mask as a uint8 GeoTIFF."""
    out_meta = meta.copy()
    out_meta.update(dtype="uint8", count=1, nodata=0)
    with rasterio.open(path, "w", **out_meta) as dst:
        dst.write(arr_u8.astype("uint8"), 1)


def burned_mask_to_feature(mask_u8: np.ndarray, transform, src_crs) -> dict | None:
    """Convert the binary burn mask into one perimeter-style vector feature.

    This supports app display and export. Burned raster cells are polygonized,
    dissolved into one footprint, and then transformed to WGS84 for web use.
    """
    if int(mask_u8.sum()) == 0:
        return None

    polys = []
    mask_bool = mask_u8.astype(bool)
    for geom, val in shapes(mask_u8.astype("uint8"), mask=mask_bool, transform=transform):
        if int(val) == 1:
            poly = shape(geom)
            if poly is not None and not poly.is_empty:
                polys.append(poly)

    if not polys:
        return None

    merged = unary_union(polys)
    if merged is None or merged.is_empty:
        return None

    geom_series = gpd.GeoSeries([merged], crs=src_crs).to_crs("EPSG:4326")
    if geom_series.empty or geom_series.iloc[0] is None or geom_series.iloc[0].is_empty:
        return None

    geom_wgs84 = geom_series.iloc[0]
    return {
        "type": "Feature",
        "properties": {"region": "full_extent"},
        "geometry": geom_wgs84.__geo_interface__,
    }


def burned_mask_to_h3_feature_collection(
    mask_u8: np.ndarray,
    transform,
    src_crs,
    h3_resolution: int = 8,
) -> dict:
    """Represent burned raster cells as H3 polygons using cell-center assignment."""
    rows, cols = np.where(mask_u8.astype(bool))
    if rows.size == 0:
        return {"type": "FeatureCollection", "features": []}

    xs, ys = xy(transform, rows, cols, offset="center")
    lons, lats = warp_transform(src_crs, "EPSG:4326", xs, ys)

    h3_cells = sorted(
        {
            h3.latlng_to_cell(float(lat), float(lon), int(h3_resolution))
            for lon, lat in zip(lons, lats)
            if math.isfinite(float(lon)) and math.isfinite(float(lat))
        }
    )

    features = []
    for cell in h3_cells:
        boundary = h3.cell_to_boundary(cell)
        coords = [[float(lon), float(lat)] for lat, lon in boundary]
        if coords and coords[0] != coords[-1]:
            coords.append(coords[0])
        features.append(
            {
                "type": "Feature",
                "properties": {
                    "h3_index": str(cell),
                    "h3_resolution": int(h3_resolution),
                    "region": "h3_cell",
                },
                "geometry": {"type": "Polygon", "coordinates": [coords]},
            }
        )

    return {"type": "FeatureCollection", "features": features}


def run_model(inputs: ModelRunInputs) -> ModelRunResult:
    """Execute one full WIPP 2.0 model run.

    High-level workflow:

    1. Read the BP raster, which defines the model grid.
    2. Warp wind and RH rasters onto that grid.
    3. Build BP- and RH-based support surfaces.
    4. Convert ignition input into seed cells on the grid.
    5. Run the raster graph traversal to produce a binary burn mask.
    6. Export the burn mask, perimeter feature, and diagnostics summary.
    """
    if inputs.run_output_dir is None:
        raise ValueError("run_output_dir is required when running the core model.")

    run_output_dir = Path(inputs.run_output_dir)
    run_output_dir.mkdir(parents=True, exist_ok=True)

    # The BP raster is the master grid for the entire analysis.
    with rasterio.open(inputs.bp_path) as src:
        bp = src.read(1)
        bp_meta = src.meta.copy()
        bp_crs = src.crs
        bp_transform = src.transform
        bp_nodata = src.nodata
        bp_shape = (src.height, src.width)

    if bp_crs is None:
        raise ValueError("BP raster has no CRS.")

    # Wind is reprojected to the BP grid so directional forcing can be applied
    # cell-by-cell during traversal.
    wind_u_bpgrid = warp_to_grid(inputs.wind_u_path, bp_meta, Resampling.bilinear)
    wind_v_bpgrid = warp_to_grid(inputs.wind_v_path, bp_meta, Resampling.bilinear)

    if inputs.rh_path is None:
        raise ValueError("RH raster is required for run_model(); provide ModelRunInputs.rh_path.")
    # RH is also brought to the BP grid and then mapped to a moisture-support
    # surface. This is a practical surrogate, not a full fuel-moisture model.
    rh_bpgrid = warp_to_grid(inputs.rh_path, bp_meta, Resampling.bilinear)

    # RH is currently a provisional moisture proxy. It suppresses local spread
    # support but does not alter path travel cost directly.
    moisture_support = build_moisture_support_from_rh(
        rh_grid=rh_bpgrid,
        params=inputs.params,
        shape_=bp_shape,
    )

    # Ignition can come from a clicked lat/lng or an uploaded feature layer.
    if inputs.ign_gdf is None:
        if inputs.ign_lat is None or inputs.ign_lng is None:
            raise ValueError("Provide either a clicked ignition point or an uploaded ignition file.")
        ign_gdf = gpd.GeoDataFrame(
            {"id": [1]},
            geometry=[Point(float(inputs.ign_lng), float(inputs.ign_lat))],
            crs="EPSG:4326",
        ).to_crs(bp_crs)
    else:
        ign_gdf = inputs.ign_gdf.to_crs(bp_crs)

    bp_float = bp.astype("float32")
    invalid_bp = ~np.isfinite(bp_float)
    if bp_nodata is not None:
        invalid_bp |= bp == bp_nodata

    # Clean the BP raster before scaling so nodata sentinels do not propagate
    # into the support/cost surfaces.
    bp_clean = np.where(invalid_bp, 0.0, bp_float)
    burn_pct = bp_clean * 100.0
    burn_pct[invalid_bp] = 0.0

    # Some workflows treat zero BP as non-burnable background. Others may want
    # to preserve it. This switch keeps that behavior explicit.
    if inputs.exclude_zero:
        burn_raster = np.where(burn_pct > 0, burn_pct, 0.0).astype("float32")
    else:
        burn_raster = burn_pct.astype("float32")

    epsilon = 1e-6
    valid_burnable = (~invalid_bp) & (burn_raster > 0.0)
    _, bp_support, base_cost = build_bp_surfaces(
        burn_raster=burn_raster,
        valid_burnable=valid_burnable,
        params=inputs.params,
        epsilon=epsilon,
    )

    seeds = points_to_seed_cells(ign_gdf, bp_transform, bp_shape)
    if not seeds:
        raise ValueError("Ignition point is outside the BP raster extent.")

    # Spread extent is determined by local support and cumulative travel cost,
    # not by imposing a fixed burn duration or a target area.
    blocked = invalid_bp
    diagnostics: dict[str, float | int | str] = {}
    burned = dijkstra_local_score_reach(
        base_cost=base_cost,
        bp_support=bp_support,
        wind_u=wind_u_bpgrid,
        wind_v=wind_v_bpgrid,
        start_rcs=seeds,
        blocked=blocked,
        use_8_neighbor=True,
        params=inputs.params,
        moisture_support=moisture_support,
        diagnostics=diagnostics,
    )

    # Export the raw burned-cell mask for downstream GIS review.
    out_raster = run_output_dir / "burnmask_wupp2b.tif"
    write_u8_raster(out_raster, burned, bp_meta)

    burned_cells = int(burned.sum())
    pixel_area_m2 = abs(float(bp_transform.a * bp_transform.e))
    # Area is derived directly from burned cell count and pixel area.
    total_area_acres = (burned_cells * pixel_area_m2) / 4046.8564224
    summary_row = {
        "region": "full_extent",
        "burnmask_path": str(out_raster),
        "burned_cells": burned_cells,
        "area_acres": total_area_acres,
        "local_score_threshold": float(inputs.params.local_score_threshold),
        "cost_decay_lambda": float(inputs.params.cost_decay_lambda),
        "spread_rate": float(inputs.params.spread_rate),
        "rh_full_support": float(inputs.params.rh_full_support),
        "rh_extinction_like": float(inputs.params.rh_extinction_like),
        "moisture_support_floor": min(1.0, max(0.0, float(inputs.params.moisture_support_floor))),
        "h3_resolution": 8,
        "stopping_rule": "local_score",
    }
    summary_row.update(diagnostics)

    # Write a flat summary table so collaborators can inspect run settings and
    # diagnostics without opening Python objects.
    summary_csv = run_output_dir / "results_summary.csv"
    summary_fieldnames = [
        "region",
        "burnmask_path",
        "burned_cells",
        "area_acres",
        "local_score_threshold",
        "cost_decay_lambda",
        "spread_rate",
        "rh_full_support",
        "rh_extinction_like",
        "moisture_support_floor",
        "h3_resolution",
        "h3_cell_count",
        "h3_geojson_path",
        "stopping_rule",
        "accepted_local_score_count",
        "accepted_local_score_q00",
        "accepted_local_score_q25",
        "accepted_local_score_q50",
        "accepted_local_score_q75",
        "accepted_local_score_q100",
        "rejected_local_score_count",
        "rejected_local_score_q00",
        "rejected_local_score_q25",
        "rejected_local_score_q50",
        "rejected_local_score_q75",
        "rejected_local_score_q100",
        "cumulative_cost_count",
        "cumulative_cost_q00",
        "cumulative_cost_q25",
        "cumulative_cost_q50",
        "cumulative_cost_q75",
        "cumulative_cost_q100",
    ]
    with summary_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=summary_fieldnames)
        writer.writeheader()
        writer.writerow({key: summary_row.get(key, "") for key in summary_fieldnames})

    if burned_cells == 0:
        raise ValueError("No burned cells were generated for the current inputs.")

    # Keep the dissolved perimeter export for GIS compatibility, but display H3
    # cells in the app so the modeled burn is represented on a stable index grid.
    feature = burned_mask_to_feature(burned, bp_transform, bp_crs)
    if feature is None:
        raise ValueError("Burned cells were generated, but the perimeter geometry could not be created. Check ignition location and raster alignment.")
    feature["properties"]["burned_cells"] = burned_cells
    feature["properties"]["area_acres"] = total_area_acres
    perimeter_geojson = {"type": "FeatureCollection", "features": [feature]}
    perimeter_path = run_output_dir / "burn_perimeter.geojson"
    with perimeter_path.open("w", encoding="utf-8") as f:
        json.dump(perimeter_geojson, f)

    h3_geojson = burned_mask_to_h3_feature_collection(
        burned,
        bp_transform,
        bp_crs,
        h3_resolution=8,
    )
    h3_path = run_output_dir / "burn_h3_r8.geojson"
    with h3_path.open("w", encoding="utf-8") as f:
        json.dump(h3_geojson, f)

    h3_cell_count = len(h3_geojson.get("features", []))
    for h3_feature in h3_geojson.get("features", []):
        h3_feature.setdefault("properties", {})["burned_cells"] = burned_cells
        h3_feature["properties"]["area_acres"] = total_area_acres
        h3_feature["properties"]["h3_cell_count"] = h3_cell_count

    summary_row["h3_cell_count"] = h3_cell_count
    summary_row["h3_geojson_path"] = str(h3_path)

    # Re-write the summary after H3 generation so the H3 diagnostics are present.
    with summary_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=summary_fieldnames)
        writer.writeheader()
        writer.writerow({key: summary_row.get(key, "") for key in summary_fieldnames})

    return ModelRunResult(
        summary=[summary_row],
        total_area_acres=total_area_acres,
        geojson=h3_geojson,
        burnmask_path=out_raster,
        summary_csv_path=summary_csv,
        burned_cells=burned_cells,
        params=inputs.params,
    )
