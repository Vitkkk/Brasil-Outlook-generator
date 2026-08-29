from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field


PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = PROJECT_ROOT / "configs"


class DomainConfig(BaseModel):
    name: str
    north: float
    south: float
    west: float
    east: float
    grid_resolution_deg: float = Field(gt=0)
    probability_radius_km: float = Field(gt=0)


class PublicationConfig(BaseModel):
    require_critical_variables: bool = True
    min_model_count: int = Field(default=2, ge=1)


class AppConfig(BaseModel):
    domain: DomainConfig
    forecast_windows: dict[str, Any]
    publication: PublicationConfig
    risk_thresholds: dict[str, Any]


def _read_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    if not isinstance(data, dict):
        raise ValueError(f"Expected mapping in {path}")
    return data


@lru_cache(maxsize=1)
def get_config() -> AppConfig:
    domain_data = _read_yaml(CONFIG_DIR / "domain.yaml")
    risk_data = _read_yaml(CONFIG_DIR / "risk_thresholds.yaml")
    return AppConfig(
        domain=DomainConfig(**domain_data["domain"]),
        forecast_windows=domain_data["forecast_windows"],
        publication=PublicationConfig(**domain_data["publication"]),
        risk_thresholds=risk_data,
    )
