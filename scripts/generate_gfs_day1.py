from __future__ import annotations

import argparse
from datetime import timedelta
import json
from pathlib import Path

from app.config import get_config
from app.hazards.gfs_proxy import aggregate_day1_max, gfs_proxy_probabilities
from app.models.gfs import GFSAdapter
from app.outlook.categories import categorical_outlook
from app.outlook.categorical_geojson import categorical_field_to_geojson
from app.outlook.geojson import probability_field_to_geojson


DEFAULT_HOURS = tuple(range(0, 25, 3))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate the first live GFS Day-1 proxy outlook")
    parser.add_argument("--data-dir", default="data/gfs", help="GRIB cache directory")
    parser.add_argument("--output-dir", default="output/gfs", help="Generated output directory")
    parser.add_argument(
        "--hours",
        default=",".join(str(x) for x in DEFAULT_HOURS),
        help="Comma-separated GFS forecast hours sampled for Day 1",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    adapter = GFSAdapter()
    cfg = get_config()
    cycle = adapter.latest_cycle()
    available = set(adapter.discover_forecast_hours(cycle))
    requested = [int(value) for value in args.hours.split(",") if value.strip()]
    hours = [hour for hour in requested if hour in available]
    if not hours:
        raise RuntimeError("None of the requested Day-1 hours are available")

    data_dir = Path(args.data_dir) / cycle.strftime("%Y%m%d%H")
    output_dir = Path(args.output_dir) / cycle.strftime("%Y%m%d%H")
    output_dir.mkdir(parents=True, exist_ok=True)

    sampled = []
    for hour in hours:
        [path] = adapter.download(cycle, [hour], data_dir)
        native = adapter.open_native([path])
        standard = adapter.standardize(native)
        sampled.append(gfs_proxy_probabilities(standard))

    fields = aggregate_day1_max(sampled)
    fields.to_netcdf(output_dir / "gfs_day1_proxy_fields.nc")

    start = cycle
    end = cycle + timedelta(hours=24)
    simplify_cfg = cfg.risk_thresholds["polygonization"]
    simplify = float(simplify_cfg["simplify_tolerance_deg"])
    min_area = float(simplify_cfg["min_area_deg2"])

    products: dict[str, dict] = {}
    hazard_thresholds = {
        "thunderstorm": [0.10, 0.40, 0.70],
        "severe": [0.05, 0.15, 0.30, 0.45, 0.60],
        "tornado": cfg.risk_thresholds["hazards"]["tornado"]["contours"],
        "hail": cfg.risk_thresholds["hazards"]["hail"]["contours"],
        "wind": cfg.risk_thresholds["hazards"]["wind"]["contours"],
    }
    for name, thresholds in hazard_thresholds.items():
        products[name] = probability_field_to_geojson(
            fields[name],
            thresholds,
            product=name,
            valid_start=start,
            valid_end=end,
            distance_km=cfg.domain.probability_radius_km,
            simplify_tolerance_deg=simplify,
            min_area_deg2=min_area,
        )
        products[name]["properties"].update(
            {
                "model": "GFS 0.25",
                "cycle": cycle.isoformat(),
                "sampled_forecast_hours": hours,
                "calibrated": False,
                "method": "GFS proxy v1",
            }
        )
        with (output_dir / f"{name}.geojson").open("w", encoding="utf-8") as handle:
            json.dump(products[name], handle, ensure_ascii=False)

    categories = categorical_outlook(fields["severe"], fields["thunderstorm"], cfg.risk_thresholds)
    categorical = categorical_field_to_geojson(
        categories,
        valid_start=start,
        valid_end=end,
        simplify_tolerance_deg=simplify,
        min_area_deg2=min_area,
    )
    categorical["properties"].update(
        {
            "model": "GFS 0.25",
            "cycle": cycle.isoformat(),
            "sampled_forecast_hours": hours,
            "calibrated": False,
            "method": "GFS proxy v1",
        }
    )
    with (output_dir / "categorical.geojson").open("w", encoding="utf-8") as handle:
        json.dump(categorical, handle, ensure_ascii=False)

    manifest = {
        "model": "GFS 0.25",
        "cycle": cycle.isoformat(),
        "valid_start": start.isoformat(),
        "valid_end": end.isoformat(),
        "sampled_forecast_hours": hours,
        "data_source": "NOAA/NCEP NOMADS",
        "calibrated": False,
        "warning": "First live-data proxy; not yet calibrated as operational probability guidance.",
    }
    with (output_dir / "manifest.json").open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2, ensure_ascii=False)

    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
