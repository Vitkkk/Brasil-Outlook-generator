from __future__ import annotations

from functools import lru_cache
import json
from pathlib import Path

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware

from .config import get_config
from .demo import build_synthetic_probability_fields, demo_valid_period
from .models.gfs import GFSAdapter
from .outlook.categories import categorical_outlook
from .outlook.categorical_geojson import categorical_field_to_geojson
from .outlook.geojson import probability_field_to_geojson
from .state import RunState


PROJECT_ROOT = Path(__file__).resolve().parents[2]
GFS_OUTPUT_ROOT = PROJECT_ROOT / "output" / "gfs"

app = FastAPI(
    title="Brazil Severe Weather Outlook API",
    version="0.2.0",
    description=(
        "Automated severe-convective outlook API. The live GFS/NOMADS adapter is now "
        "available; if no generated live product exists on disk, public outlook endpoints "
        "fall back to the explicitly-labelled synthetic MVP field."
    ),
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["GET"],
    allow_headers=["*"],
)


@lru_cache(maxsize=1)
def _demo_fields():
    return build_synthetic_probability_fields()


def _polygon_config() -> tuple[float, float]:
    cfg = get_config().risk_thresholds["polygonization"]
    return float(cfg["simplify_tolerance_deg"]), float(cfg["min_area_deg2"])


def _latest_live_file(filename: str) -> Path | None:
    if not GFS_OUTPUT_ROOT.exists():
        return None
    cycle_dirs = sorted(
        (path for path in GFS_OUTPUT_ROOT.iterdir() if path.is_dir()),
        key=lambda path: path.name,
        reverse=True,
    )
    for cycle_dir in cycle_dirs:
        candidate = cycle_dir / filename
        if candidate.is_file():
            return candidate
    return None


def _read_latest_live_geojson(filename: str) -> dict | None:
    path = _latest_live_file(filename)
    if path is None:
        return None
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    payload.setdefault("properties", {})["served_from"] = str(path.relative_to(PROJECT_ROOT))
    return payload


def _hazard_geojson(product: str) -> dict:
    live = _read_latest_live_geojson(f"{product}.geojson")
    if live is not None:
        return live

    cfg = get_config()
    ds = _demo_fields()
    start, end = demo_valid_period()

    if product == "thunderstorm":
        thresholds = [0.10, 0.40, 0.70]
    elif product == "severe":
        thresholds = [0.05, 0.15, 0.30, 0.45, 0.60]
    elif product in {"tornado", "hail", "wind"}:
        thresholds = cfg.risk_thresholds["hazards"][product]["contours"]
    else:
        raise HTTPException(status_code=404, detail=f"Unknown product: {product}")

    simplify, min_area = _polygon_config()
    result = probability_field_to_geojson(
        ds[product],
        thresholds,
        product=product,
        valid_start=start,
        valid_end=end,
        distance_km=cfg.domain.probability_radius_km,
        simplify_tolerance_deg=simplify,
        min_area_deg2=min_area,
    )
    result["properties"].update(
        {
            "data_source": "synthetic_mvp",
            "operational": False,
            "warning": "Synthetic demonstration field; not a real weather forecast.",
        }
    )
    return result


def _categorical_geojson() -> dict:
    live = _read_latest_live_geojson("categorical.geojson")
    if live is not None:
        return live

    cfg = get_config()
    ds = _demo_fields()
    start, end = demo_valid_period()
    categories = categorical_outlook(ds["severe"], ds["thunderstorm"], cfg.risk_thresholds)
    simplify, min_area = _polygon_config()
    result = categorical_field_to_geojson(
        categories,
        valid_start=start,
        valid_end=end,
        simplify_tolerance_deg=simplify,
        min_area_deg2=min_area,
    )
    result["properties"].update(
        {
            "data_source": "synthetic_mvp",
            "operational": False,
            "warning": "Synthetic demonstration field; not a real weather forecast.",
        }
    )
    return result


def _gfs_status_payload() -> dict:
    adapter = GFSAdapter()
    try:
        cycle = adapter.latest_cycle()
        hours = adapter.discover_forecast_hours(cycle)
        return {
            "configured": True,
            "connected": True,
            "source": "NOAA/NCEP NOMADS GFS 0.25 degree",
            "latest_cycle": cycle.isoformat(),
            "available_forecast_hours": hours,
            "available_hour_count": len(hours),
            "state": RunState.WAITING_FOR_DATA,
        }
    except Exception as exc:
        return {
            "configured": True,
            "connected": False,
            "source": "NOAA/NCEP NOMADS GFS 0.25 degree",
            "state": RunState.FAILED,
            "error": str(exc),
        }


@app.get("/")
def root() -> dict:
    return {
        "name": "Brazil Severe Weather Outlook API",
        "version": "0.2.0",
        "status": "LIVE_GFS_INGESTION_MVP",
        "docs": "/docs",
    }


@app.get("/health")
def health() -> dict:
    return {"ok": True}


@app.get("/api/models/status")
def model_status() -> dict:
    live_product = _latest_live_file("manifest.json")
    return {
        "state": RunState.READY if live_product else RunState.WAITING_FOR_DATA,
        "models": {
            "GFS": _gfs_status_payload(),
            "GEFS": {"configured": False, "connected": False},
            "ECMWF": {"configured": False, "connected": False},
            "EPS": {"configured": False, "connected": False},
            "ICON": {"configured": False, "connected": False},
            "WRF": {"configured": False, "connected": False},
        },
        "live_gfs_product_available": live_product is not None,
        "synthetic_fallback_enabled": True,
    }


@app.get("/api/models/gfs/status")
def gfs_status() -> dict:
    return _gfs_status_payload()


@app.get("/api/models/gfs/subset-url")
def gfs_subset_url(forecast_hour: int = Query(default=12, ge=0, le=384)) -> dict:
    adapter = GFSAdapter()
    cycle = adapter.latest_cycle()
    available = adapter.discover_forecast_hours(cycle)
    if forecast_hour not in available:
        raise HTTPException(
            status_code=404,
            detail=f"Forecast hour f{forecast_hour:03d} is not available for {cycle:%Y%m%d%H}",
        )
    return {
        "model": "GFS 0.25",
        "cycle": cycle.isoformat(),
        "forecast_hour": forecast_hour,
        "url": adapter.subset_url(cycle, forecast_hour),
    }


@app.get("/api/outlook/latest")
def latest_outlook() -> dict:
    return _categorical_geojson()


@app.get("/api/outlook/day1")
def day1_outlook() -> dict:
    return _categorical_geojson()


@app.get("/api/outlook/day2")
def day2_outlook() -> dict:
    raise HTTPException(status_code=501, detail="Day 2 pipeline is not implemented yet")


@app.get("/api/outlook/day3")
def day3_outlook() -> dict:
    raise HTTPException(status_code=501, detail="Day 3 pipeline is not implemented yet")


@app.get("/api/hazard/tornado")
def tornado_outlook() -> dict:
    return _hazard_geojson("tornado")


@app.get("/api/hazard/hail")
def hail_outlook() -> dict:
    return _hazard_geojson("hail")


@app.get("/api/hazard/wind")
def wind_outlook() -> dict:
    return _hazard_geojson("wind")


@app.get("/api/hazard/severe")
def severe_outlook() -> dict:
    return _hazard_geojson("severe")


@app.get("/api/hazard/thunderstorm")
def thunderstorm_outlook() -> dict:
    return _hazard_geojson("thunderstorm")
