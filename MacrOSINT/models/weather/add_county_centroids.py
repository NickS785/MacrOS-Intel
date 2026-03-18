# save as: add_county_centroids.py
import io, os, zipfile, requests, pandas as pd
from pathlib import Path
import geopandas as gpd, pandas as pd

GAZ_BASE = "http://www2.census.gov/geo/docs/maps-data/data/gazetteer/{year}_Gazetteer"
GAZ_NATIONAL = "{year}_Gaz_counties_national.zip"  # contains {year}_gaz_counties_national.txt

def _cache_path(fname: str) -> Path:
    cache = Path.home() / ".cache" / "us_gazetteer"
    cache.mkdir(parents=True, exist_ok=True)
    return cache / fname

def fetch_gazetteer_counties(year: int = 2024) -> pd.DataFrame:
    """
    Download (once) and parse the national Counties Gazetteer.
    Columns include: GEOID, NAME, INTPTLAT, INTPTLONG (lat/lon in decimal degrees).
    """
    url = f"{GAZ_BASE.format(year=year)}/{GAZ_NATIONAL.format(year=year)}"
    zpath = _cache_path(GAZ_NATIONAL.format(year=year))
    if not zpath.exists():
        r = requests.get(url, timeout=60)
        r.raise_for_status()
        zpath.write_bytes(r.content)

    with zipfile.ZipFile(zpath, "r") as zf:
        # find the single .txt inside (e.g., 2024_gaz_counties_national.txt)
        txt_name = [n for n in zf.namelist() if n.endswith(".txt")][0]
        with zf.open(txt_name) as f:
            gaz = pd.read_csv(f, sep="\t", dtype={"GEOID": str})
    gaz.columns = gaz.columns.str.strip()
    # Normalize lat/lon as float
    gaz["INTPTLAT"] = gaz["INTPTLAT"].astype(str).str.strip().astype(float)
    gaz["INTPTLONG"] = gaz["INTPTLONG"].astype(str).str.strip().astype(float)
    return gaz[["GEOID", "NAME", "INTPTLAT", "INTPTLONG"]].rename(
        columns={"INTPTLAT": "latitude", "INTPTLONG": "longitude"}
    )

def build_geoid5(df: pd.DataFrame,
                 state_col: str = "state_fips_code",
                 county_col_candidates=("county_code_clean","county_code")) -> pd.Series:
    if state_col not in df.columns:
        raise KeyError(f"Missing column: {state_col}")
    county_col = next((c for c in county_col_candidates if c in df.columns), None)
    if county_col is None:
        raise KeyError(f"Missing county code column; tried {county_col_candidates}")
    return (
        df[state_col].astype(str).str.zfill(2) +
        df[county_col].astype(str).str.zfill(3)
    )

def append_centroids_via_shapes(selection, out_csv: str | None = None):
    if isinstance(selection, str):
        sel = pd.read_csv(selection, dtype=str)
    else:
        sel = selection
    sel["GEOID"] = (
        sel["state_fips_code"].astype(str).str.zfill(2) +
        sel[["county_code_clean","county_code"]]
          .apply(lambda s: (s.dropna().astype(str)).iloc[0] if s.notna().any() else "", axis=1)
          .str.zfill(3)
    )

    # 2023 counties (1:500k)
    
    shp_url = "C:\\Users\\nicho\PycharmProjects\WeatherReport\counties\cb_2023_us_county_500k.shp"
    g = gpd.read_file(shp_url).to_crs(4326)
    g["GEOID"] = g["GEOID"].astype(str)
    g["latitude"]  = g.representative_point().y
    g["longitude"] = g.representative_point().x

    out = sel.merge(g[["GEOID","latitude","longitude"]], on="GEOID", how="left")
    if out_csv is None and isinstance(selection, str):
        if selection.endswith('.csv'):
            out_csv = selection.replace(".csv","_with_latlon.csv")
        else:
            out_csv = 'selected_counties_for_weather.csv'

    out.to_csv(out_csv, index=False)
    print(f"Saved: {out_csv}")
    return out

