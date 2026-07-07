# WIPP 2.0

Wildfire Ignition Perimeter Projection (WIPP) is a local Flask app that projects wildfire spread from an ignition point using:

- a static Tennessee burn-probability raster,
- Open-Meteo wind U/V rasters,
- Open-Meteo relative humidity,
- a calibrated local spread-sustainability score.

The model runs on the BP raster grid and displays burned output as H3 resolution 8 cells.

## Included Runtime Data

This repository includes the app code, core model code, model defaults, Tennessee BP raster, and cached Open-Meteo raster inputs needed to run the app locally.

Large calibration datasets and generated calibration outputs are intentionally excluded from Git.

## Run Locally

```powershell
py -m pip install -r requirements.txt
py app.py
```

Then open:

```text
http://127.0.0.1:5000
```

## Main Files

- `app.py` - Flask app and API routes
- `index.html` - browser UI
- `wipp_core.py` - core spread model
- `build_openmeteo_rasters.py` - Open-Meteo raster builder
- `model_defaults.json` - current calibrated Tennessee parameter defaults
- `BP/Tennessee_BP.tif` - static burn-probability raster
- `openmeteo_windu_rast/` - cached wind U raster
- `openmeteo_windv_rast/` - cached wind V raster
- `openmeteo_rh_rast/` - cached relative humidity raster
