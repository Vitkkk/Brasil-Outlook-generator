from datetime import datetime, timezone

from app.ingestion.nomads import DomainBox, build_gfs_filter_url


def test_gfs_subset_url_contains_cycle_domain_variables_and_levels():
    cycle = datetime(2026, 8, 29, 12, tzinfo=timezone.utc)
    url = build_gfs_filter_url(
        cycle,
        12,
        variables=["CAPE", "HLCY", "UGRD", "VGRD"],
        levels=["surface", "500_mb", "3000-0_m_above_ground"],
        domain=DomainBox(north=15, south=-60, west=-90, east=-25),
    )

    assert "file=gfs.t12z.pgrb2.0p25.f012" in url
    assert "var_CAPE=on" in url
    assert "var_HLCY=on" in url
    assert "lev_surface=on" in url
    assert "lev_500_mb=on" in url
    assert "leftlon=270" in url
    assert "rightlon=335" in url
    assert "toplat=15" in url
    assert "bottomlat=-60" in url
    assert "dir=%2Fgfs.20260829%2F12%2Fatmos" in url
