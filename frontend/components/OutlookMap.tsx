"use client";

import { useEffect, useRef, useState } from "react";
import maplibregl, { Map as MapLibreMap } from "maplibre-gl";

const API_BASE = process.env.NEXT_PUBLIC_API_BASE ?? "http://localhost:8000";

const CATEGORY_STYLES = [
  { name: "TSTM", fill: "#b8e3b2", line: "#4c8b49" },
  { name: "MRGL", fill: "#67c26f", line: "#217a32" },
  { name: "SLGT", fill: "#f4e66a", line: "#b7a600" },
  { name: "ENH", fill: "#f1ad52", line: "#c87819" },
  { name: "MDT", fill: "#e9635d", line: "#a92c2a" },
  { name: "HIGH", fill: "#c96ac5", line: "#8f2f8b" },
] as const;

export default function OutlookMap() {
  const containerRef = useRef<HTMLDivElement | null>(null);
  const mapRef = useRef<MapLibreMap | null>(null);
  const [status, setStatus] = useState("Carregando outlook...");

  useEffect(() => {
    if (!containerRef.current || mapRef.current) return;

    const map = new maplibregl.Map({
      container: containerRef.current,
      style: "https://demotiles.maplibre.org/style.json",
      center: [-52.5, -20.5],
      zoom: 3.2,
      attributionControl: true,
    });
    mapRef.current = map;

    map.addControl(new maplibregl.NavigationControl(), "top-right");

    map.on("load", async () => {
      try {
        const response = await fetch(`${API_BASE}/api/outlook/day1`, { cache: "no-store" });
        if (!response.ok) throw new Error(`API returned ${response.status}`);
        const geojson = await response.json();

        map.addSource("categorical-outlook", {
          type: "geojson",
          data: geojson,
        });

        for (const style of CATEGORY_STYLES) {
          map.addLayer({
            id: `risk-fill-${style.name}`,
            type: "fill",
            source: "categorical-outlook",
            filter: ["==", ["get", "category"], style.name],
            paint: {
              "fill-color": style.fill,
              "fill-opacity": 0.48,
            },
          });
          map.addLayer({
            id: `risk-line-${style.name}`,
            type: "line",
            source: "categorical-outlook",
            filter: ["==", ["get", "category"], style.name],
            paint: {
              "line-color": style.line,
              "line-width": 2.2,
            },
          });
        }

        const warning = geojson?.properties?.warning;
        setStatus(warning ?? "Outlook carregado");
      } catch (error) {
        setStatus(`Falha ao carregar o backend: ${String(error)}`);
      }
    });

    return () => {
      map.remove();
      mapRef.current = null;
    };
  }, []);

  return (
    <section className="map-shell">
      <div ref={containerRef} className="map" />
      <div className="map-status">{status}</div>
      <div className="legend" aria-label="Legenda de risco">
        {CATEGORY_STYLES.map((item) => (
          <div className="legend-item" key={item.name}>
            <span className="legend-swatch" style={{ background: item.fill, borderColor: item.line }} />
            <span>{item.name}</span>
          </div>
        ))}
      </div>
    </section>
  );
}
