from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.pyplot as plt
from matplotlib.patches import Patch, Rectangle, Polygon
import matplotlib.patheffects as pe
import numpy as np
import xarray as xr

from app.config import get_config
from app.outlook.categories import categorical_outlook
from app.outlook.land_mask import mask_dataarray_to_land
from app.outlook.presentation import coherent_mask

# Automatic "PREVOTS-style" presentation mode. This is intentionally branded as
# AUTOMATIC / NON-OFFICIAL and does not use the PREVOTS logo.
LEVEL_COLORS = {
    1: "#b8eea5",  # Tempestades
    2: "#fff20a",  # Nivel 1
    3: "#f6ad38",  # Nivel 2
    4: "#ef4034",  # Nivel 3
    5: "#ee14df",  # Nivel 4
}
LEVEL_NAMES = {
    1: "Tempestades",
    2: "Nível 1",
    3: "Nível 2",
    4: "Nível 3",
    5: "Nível 4",
}

CITIES = [
    ("Cuiabá",-15.60,-56.10),("Brasília",-15.79,-47.88),("Goiânia",-16.68,-49.25),
    ("Montes Claros",-16.73,-43.86),("Porto Seguro",-16.44,-39.06),("Corumbá",-19.01,-57.65),
    ("Coxim",-18.51,-54.76),("Jataí",-17.88,-51.72),("Uberlândia",-18.91,-48.28),
    ("Belo Horizonte",-19.92,-43.94),("Vitória",-20.32,-40.34),("Campo Grande",-20.47,-54.62),
    ("Três Lagoas",-20.75,-51.68),("Ribeirão Preto",-21.18,-47.81),("Juiz de Fora",-21.76,-43.35),
    ("Dourados",-22.22,-54.81),("Bauru",-22.32,-49.07),("Rio de Janeiro",-22.91,-43.17),
    ("Maringá",-23.42,-51.94),("São Paulo",-23.55,-46.63),("Cascavel",-24.96,-53.46),
    ("Guarapuava",-25.39,-51.46),("Ilha Comprida",-24.74,-47.54),("Curitiba",-25.43,-49.27),
    ("Joinville",-26.30,-48.85),("Chapecó",-27.10,-52.62),("Lages",-27.82,-50.33),
    ("Florianópolis",-27.59,-48.55),("Criciúma",-28.68,-49.37),("Santa Rosa",-27.87,-54.48),
    ("Uruguaiana",-29.76,-57.09),("Santa Maria",-29.69,-53.81),("Porto Alegre",-30.03,-51.23),
    ("Pelotas",-31.77,-52.34),
]

def xy(da):
    lon=np.asarray(da.longitude); lat=np.asarray(da.latitude)
    if lon.ndim==1 and lat.ndim==1:
        return np.meshgrid(lon,lat)
    return lon,lat

def auto_levels(ds,cfg):
    cat = categorical_outlook(ds.severe, ds.thunderstorm, cfg.risk_thresholds).astype(float)
    cat = mask_dataarray_to_land(cat)
    a = np.asarray(cat)
    # Compress SPC-like six codes into the five-level PREVOTS-like visual scale:
    # TSTM -> Tempestades; MRGL -> L1; SLGT -> L2; ENH -> L3; MDT/HIGH -> L4.
    out=np.full_like(a,np.nan,dtype=float)
    out[np.isfinite(a) & (a>=1)] = 1
    out[np.isfinite(a) & (a>=2)] = 2
    out[np.isfinite(a) & (a>=3)] = 3
    out[np.isfinite(a) & (a>=4)] = 4
    out[np.isfinite(a) & (a>=5)] = 5
    return xr.DataArray(out,coords=cat.coords,dims=cat.dims)

def draw_city(ax,name,lat,lon):
    ax.scatter(lon,lat,s=62,facecolor="white",edgecolor="#222",linewidth=.9,
               transform=ccrs.PlateCarree(),zorder=12)
    t=ax.text(lon,lat-0.38,name,ha="center",va="top",fontsize=8.8,color="#111",
              transform=ccrs.PlateCarree(),zorder=13)
    t.set_path_effects([pe.withStroke(linewidth=2.4,foreground="white")])

def draw_scale(ax):
    # visual 0-100-200 km scale near southern Brazil; approximate geodesic width
    y=-32.15; x=-49.0; dlon=2.15
    ax.add_patch(Rectangle((x,y),dlon/2,.38,facecolor="black",edgecolor="black",
                           transform=ccrs.PlateCarree(),zorder=20))
    ax.add_patch(Rectangle((x+dlon/2,y),dlon/2,.38,facecolor="white",edgecolor="black",
                           transform=ccrs.PlateCarree(),zorder=20))
    ax.text(x,y+.58,"0",fontsize=9,ha="center",transform=ccrs.PlateCarree(),zorder=21)
    ax.text(x+dlon/2,y+.58,"100",fontsize=9,ha="center",transform=ccrs.PlateCarree(),zorder=21)
    ax.text(x+dlon,y+.58,"200 km",fontsize=9,ha="center",transform=ccrs.PlateCarree(),zorder=21)

def draw_north_arrow(ax):
    x,y=.945,.935
    outer=np.array([[x,y],[x-.025,y-.075],[x,y-.045],[x+.025,y-.075]])
    inner=np.array([[x,y-.006],[x,y-.045],[x+.018,y-.062]])
    ax.add_patch(Polygon(outer,closed=True,facecolor="white",edgecolor="black",lw=1.4,
                         transform=ax.transAxes,zorder=30))
    ax.add_patch(Polygon(inner,closed=True,facecolor="#c8c8c8",edgecolor="black",lw=.8,
                         transform=ax.transAxes,zorder=31))

def main():
    p=argparse.ArgumentParser()
    p.add_argument("netcdf")
    p.add_argument("output")
    p.add_argument("--manifest",required=True)
    p.add_argument("--model",default="CONSENSUS")
    a=p.parse_args()

    ds=xr.open_dataset(a.netcdf)
    cfg=get_config()
    manifest=json.loads(Path(a.manifest).read_text())
    levels=auto_levels(ds,cfg)
    lon,lat=xy(levels)
    val=np.asarray(levels)

    fig=plt.figure(figsize=(10.4,9.2),dpi=170,facecolor="white")
    ax=fig.add_axes([.035,.085,.91,.82],projection=ccrs.PlateCarree())
    ax.set_extent([-61.3,-37.0,-33.9,-14.2],crs=ccrs.PlateCarree())

    ax.add_feature(cfeature.OCEAN.with_scale("50m"),facecolor="#dfeaf4",zorder=0)
    ax.add_feature(cfeature.LAND.with_scale("50m"),facecolor="#f2f0ed",zorder=0)
    ax.add_feature(cfeature.LAKES.with_scale("50m"),facecolor="#dfeaf4",edgecolor="#b9c8d4",lw=.3,zorder=1)

    # faint terrain-like wash without external tiles
    try:
        ax.stock_img()
    except Exception:
        pass

    # Draw nested severity polygons.
    for lev in range(1,6):
        m=np.isfinite(val)&(val>=lev)
        m=coherent_mask(m,closing_iterations=2 if lev>=4 else 1,min_component_cells=1 if lev>=4 else 3)
        if not m.any():
            continue
        ax.contourf(lon,lat,np.where(m,1,np.nan),levels=[.5,1.5],
                    colors=[LEVEL_COLORS[lev]],alpha=.62,transform=ccrs.PlateCarree(),zorder=3+lev)
        ax.contour(lon,lat,m.astype(float),levels=[.5],colors=[LEVEL_COLORS[lev]],
                   linewidths=1.2,transform=ccrs.PlateCarree(),zorder=4+lev)

    ax.add_feature(cfeature.COASTLINE.with_scale("50m"),linewidth=.65,edgecolor="#555",zorder=11)
    ax.add_feature(cfeature.BORDERS.with_scale("50m"),linewidth=.75,edgecolor="#555",zorder=11)
    try:
        states=cfeature.NaturalEarthFeature("cultural","admin_1_states_provinces_lines","50m",facecolor="none")
        ax.add_feature(states,linewidth=.55,edgecolor="#666",zorder=11)
    except Exception:
        pass

    gl=ax.gridlines(draw_labels=True,linewidth=.3,color="#9fa8b0",alpha=.45,x_inline=False,y_inline=False)
    gl.top_labels=False; gl.left_labels=False
    gl.xlabel_style={"size":7.4}; gl.ylabel_style={"size":7.4,"rotation":90}

    for name,clat,clon in CITIES:
        if -61.3<clon<-37 and -33.9<clat<-14.2:
            draw_city(ax,name,clat,clon)

    draw_north_arrow(ax)
    draw_scale(ax)

    # Neighbor-country labels
    for text_,x,y in [("Bolívia",-60.3,-18.1),("Paraguai",-58.7,-23.0),("Argentina",-59.0,-28.7),("Uruguai",-56.5,-33.0)]:
        ax.text(x,y,text_,fontsize=9,fontweight="bold",color="#666",transform=ccrs.PlateCarree(),zorder=15)

    # Legend box matching the visual hierarchy of the supplied examples.
    handles=[Patch(facecolor=LEVEL_COLORS[i],edgecolor=LEVEL_COLORS[i],label=LEVEL_NAMES[i]) for i in range(1,6)]
    leg=ax.legend(handles=handles,loc="lower right",bbox_to_anchor=(.986,.115),fontsize=10.8,
                  title="Níveis de Severidade",title_fontsize=13.0,frameon=True,fancybox=False,
                  facecolor="white",edgecolor="#222",framealpha=.96,borderpad=.6,
                  handlelength=1.4,handletextpad=.5,labelspacing=.35)
    leg.get_title().set_fontweight("bold")

    valid=manifest.get("valid_start",manifest.get("cycle",""))
    try:
        date=datetime.fromisoformat(valid.replace("Z","+00:00")).strftime("%d/%m/%Y")
    except Exception:
        date="29/09/2026"

    fig.suptitle(f"PREVISÃO AUTOMÁTICA DE TEMPO SEVERO - {date} (*)",
                 fontsize=19.5,fontweight="bold",y=.962)
    fig.text(.035,.035,"(*) Produto automático experimental — NÃO OFICIAL.",fontsize=10.6,fontweight="bold")
    fig.text(.945,.035,"Modo PREVOTS-style • GFS + ECMWF",fontsize=8.4,ha="right",color="#555")

    # Auto badge instead of PREVOTS branding/logo.
    fig.text(.842,.115,"AUTO\nSEVERE",ha="center",va="center",fontsize=18,fontweight="bold",
             bbox=dict(boxstyle="square,pad=.45",facecolor="white",edgecolor="#222",linewidth=1.2))

    out=Path(a.output)
    out.parent.mkdir(parents=True,exist_ok=True)
    fig.savefig(out,facecolor="white",bbox_inches="tight")
    plt.close(fig)
    print(out)

if __name__=="__main__":
    main()
