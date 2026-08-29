# NOMADS GFS subset links used by the ingestion adapter

This file documents reproducible NOMADS grib-filter requests for the South America domain used by the project.

## 2026-08-29 12Z — f012 test subset

[Download f012 convective subset](https://nomads.ncep.noaa.gov/cgi-bin/filter_gfs_0p25.pl?file=gfs.t12z.pgrb2.0p25.f012&var_CAPE=on&var_CIN=on&var_DPT=on&var_GUST=on&var_HGT=on&var_HLCY=on&var_PWAT=on&var_REFC=on&var_RH=on&var_TMP=on&var_UGRD=on&var_USTM=on&var_VGRD=on&var_VSTM=on&var_VVEL=on&lev_surface=on&lev_2_m_above_ground=on&lev_10_m_above_ground=on&lev_3000-0_m_above_ground=on&lev_6000-0_m_above_ground=on&lev_180-0_mb_above_ground=on&lev_90-0_mb_above_ground=on&lev_1000_mb=on&lev_925_mb=on&lev_850_mb=on&lev_700_mb=on&lev_500_mb=on&lev_300_mb=on&lev_250_mb=on&lev_entire_atmosphere_%28considered_as_a_single_layer%29=on&subregion=&leftlon=270&rightlon=335&toplat=15&bottomlat=-60&dir=%2Fgfs.20260829%2F12%2Fatmos)

The operational adapter must generate these URLs programmatically rather than hard-code this date/cycle.