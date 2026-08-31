from __future__ import annotations

import argparse
import json
from pathlib import Path

import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np
import xarray as xr

from app.config import get_config
from app.outlook.categories import categorical_outlook
from app.outlook.land_mask import mask_dataarray_to_land
from app.outlook.presentation import coherent_mask

CAT_COLORS={1:"#cdeec8",2:"#58b957",3:"#f5e84a",4:"#f3a43a",5:"#e85b55",6:"#d94fd5"}
CAT_NAMES={1:"TSTM",2:"MRGL",3:"SLGT",4:"ENH",5:"MDT",6:"HIGH"}
HAZARD_COLORS={0.02:"#58b957",0.05:"#9f6b4a",0.10:"#f0cf32",0.15:"#f0a333",0.30:"#e65345",0.45:"#8853d1",0.60:"#111111"}


def setup(ax,left,bottom):
    ax.set_extent([-68,-34,-42,-14],crs=ccrs.PlateCarree())
    ax.add_feature(cfeature.LAND.with_scale("50m"),facecolor="white",zorder=0)
    ax.add_feature(cfeature.OCEAN.with_scale("50m"),facecolor="#f5f6f7",zorder=0)
    ax.add_feature(cfeature.COASTLINE.with_scale("50m"),linewidth=.65,edgecolor="#555",zorder=5)
    ax.add_feature(cfeature.BORDERS.with_scale("50m"),linewidth=.65,edgecolor="#666",zorder=5)
    try:
        states=cfeature.NaturalEarthFeature("cultural","admin_1_states_provinces_lines","50m",facecolor="none")
        ax.add_feature(states,linewidth=.4,edgecolor="#888",zorder=5)
    except Exception: pass
    gl=ax.gridlines(draw_labels=True,linewidth=.25,color="#c8ccd2",alpha=.65)
    gl.top_labels=gl.right_labels=False; gl.left_labels=left; gl.bottom_labels=bottom
    gl.xlabel_style=gl.ylabel_style={"size":7}


def xy(da):
    lon=np.asarray(da.longitude); lat=np.asarray(da.latitude)
    if lon.ndim==1 and lat.ndim==1: return np.meshgrid(lon,lat)
    return lon,lat


def clean(mask,strong=False):
    return coherent_mask(mask,closing_iterations=2 if strong else 1,min_component_cells=1 if strong else 3)


def category(ax,ds,cfg):
    cat=mask_dataarray_to_land(categorical_outlook(ds.severe,ds.thunderstorm,cfg.risk_thresholds).astype(float))
    lon,lat=xy(cat); val=np.asarray(cat); reached=[]
    for code in range(1,7):
        m=clean(np.isfinite(val)&(val>=code),code>=4)
        if not m.any(): continue
        reached.append(code)
        ax.contourf(lon,lat,np.where(m,1,np.nan),levels=[.5,1.5],colors=[CAT_COLORS[code]],alpha=.72,transform=ccrs.PlateCarree())
        ax.contour(lon,lat,m.astype(float),levels=[.5],colors=[CAT_COLORS[code]],linewidths=1.2,transform=ccrs.PlateCarree())
    if reached:
        ax.legend([Patch(facecolor=CAT_COLORS[c],edgecolor="#555") for c in reached],[CAT_NAMES[c] for c in reached],loc="lower right",ncol=3,fontsize=7,title="Risk Category",title_fontsize=8,framealpha=.95)


def hazard(ax,da,thresholds):
    da=mask_dataarray_to_land(da); lon,lat=xy(da); val=np.asarray(da); reached=[]
    for t in thresholds:
        m=clean(np.isfinite(val)&(val>=t),t>=.30)
        if not m.any(): continue
        reached.append(t)
        ax.contourf(lon,lat,np.where(m,1,np.nan),levels=[.5,1.5],colors=[HAZARD_COLORS[t]],alpha=.66,transform=ccrs.PlateCarree())
        ax.contour(lon,lat,m.astype(float),levels=[.5],colors=[HAZARD_COLORS[t]],linewidths=1.15,transform=ccrs.PlateCarree())
    if reached:
        ax.legend([Patch(facecolor=HAZARD_COLORS[t],edgecolor="#555") for t in reached],[f"{int(t*100)}%" for t in reached],loc="lower right",ncol=2,fontsize=7,title="Probability within 40 km",title_fontsize=8,framealpha=.95)


def main():
    p=argparse.ArgumentParser(); p.add_argument("netcdf"); p.add_argument("output"); p.add_argument("--manifest",required=True); p.add_argument("--model",required=True)
    a=p.parse_args(); ds=xr.open_dataset(a.netcdf); cfg=get_config(); m=json.loads(Path(a.manifest).read_text())
    fig=plt.figure(figsize=(14,14),dpi=155,facecolor="white"); gs=fig.add_gridspec(2,2,left=.055,right=.955,bottom=.095,top=.815,wspace=.10,hspace=.12)
    products=[("CATEGORICAL CONVECTIVE OUTLOOK",None,None),("TORNADO PROBABILITY","tornado",[.02,.05,.10,.15,.30,.45,.60]),("HAIL PROBABILITY","hail",[.02,.05,.15,.30,.45,.60]),("WIND PROBABILITY","wind",[.02,.05,.15,.30,.45,.60])]
    for i,(title,name,ths) in enumerate(products):
        r,c=divmod(i,2); ax=fig.add_subplot(gs[r,c],projection=ccrs.PlateCarree()); setup(ax,c==0,r==1)
        category(ax,ds,cfg) if name is None else hazard(ax,ds[name],ths)
        b=ax.get_position(); fig.text((b.x0+b.x1)/2,b.y1+.010,title,ha="center",va="bottom",fontsize=11.4,fontweight="bold")
    mx=m["maxima"][a.model]
    fig.suptitle(f"BRAZIL SEVERE WEATHER OUTLOOK — {a.model}",fontsize=20.5,fontweight="bold",y=.955)
    fig.text(.5,.920,f"CYCLE {m['cycle']} • VALID {m['valid_start']} – {m['valid_end']}",ha="center",fontsize=10.2)
    note="DETERMINISTIC V3 • LAND-ONLY • VERTICAL SOUNDING DIAGNOSTICS"
    if "WRF" in a.model.upper(): note += " • MESOSCALE WRF 7-km"
    fig.text(.5,.887,note,ha="center",fontsize=8.5,fontweight="bold",bbox=dict(boxstyle="round,pad=.34",facecolor="#fff0b8",edgecolor="#987a14"))
    fig.text(.5,.852,f"MAXIMA: severe {mx['severe']*100:.1f}% • tornado {mx['tornado']*100:.1f}% • hail {mx['hail']*100:.1f}% • wind {mx['wind']*100:.1f}%",ha="center",fontsize=8.5,color="#444")
    footer="Engineering guidance; probabilities are not yet historically calibrated. WRF reflectivity contributes to initiation only when present in the source GRIB."
    fig.text(.055,.040,footer,fontsize=7.6,color="#50545a",wrap=True)
    out=Path(a.output); out.parent.mkdir(parents=True,exist_ok=True); fig.savefig(out,facecolor="white"); plt.close(fig); print(out)

if __name__=="__main__": main()
