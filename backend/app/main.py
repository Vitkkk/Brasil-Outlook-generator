from __future__ import annotations

from functools import lru_cache

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from .config import get_config
from .demo import build_synthetic_probability_fields, demo_valid_period
from .outlook.categories import categorical_outlook
from .outlook.categorical_geojson import categorical_field_to_geojson
from .outlook.geojson import probability_field_to_geojson
from .state import RunState


app = FastAPI(
    title="Brazil Severe Weather Outlook API",
    version="0.1.0",
    description=(
        "MVP API for an automated severe-convective outlook system. "
        "Current probability grids are synthetic until live model adapters are connected."
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


def _hazard_geojson(product: str) -> dict:
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


@app.get("/")
def root() -> dict:
    return {
        "name": "Brazil Severe Weather Outlook API",
        "version": "0.1.0",
        "status": "MVP",
        "docs": "/docs",
    }


@app.get("/health")
def health() -> dict:
    return {"ok": True}


@app.get("/api/models/status")
def model_status() -> dict:
    return {
        "state": RunState.WAITING_FOR_DATA,
        "models": {
            "GFS": {"connected": False},
            "GEFS": {"connected": False},
            "ECMWF": {"connected": False},
            "EPS": {"connected": False},
            "ICON": {"connected": False},
            "WRF": {"connected": False},
        },
        "demo_mode": True,
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
