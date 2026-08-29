import OutlookMap from "../components/OutlookMap";

export default function HomePage() {
  return (
    <main className="page">
      <header className="topbar">
        <div>
          <p className="eyebrow">BRAZIL SEVERE WEATHER OUTLOOK</p>
          <h1>Day 1 Convective Outlook</h1>
          <p className="subtitle">Probabilistic severe-weather guidance for Brazil and adjacent South America</p>
        </div>
        <div className="badge">MVP / DEMO DATA</div>
      </header>

      <OutlookMap />

      <section className="notice">
        The current repository build uses synthetic probability grids only. Live GFS/GEFS/ECMWF/ICON/WRF ingestion and calibrated hazard probabilities are the next implementation stage.
      </section>
    </main>
  );
}
