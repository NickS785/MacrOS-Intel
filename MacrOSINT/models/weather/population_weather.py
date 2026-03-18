"""
Population-weighted weather grid.

Clusters US metro-area counties by population density, locates nearby
GHCND stations via DensityBasedLocator, and produces population-weighted
daily/monthly temperature and precipitation averages at state or national level.
"""
import json
from datetime import date, timedelta
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd

from .add_county_centroids import fetch_gazetteer_counties
from .agclimate import AgClimateAPI
from .density_station_locator import (
    DensityBasedLocator,
    DensityCluster,
    StationMatch,
)

CENSUS_DIR = Path(__file__).resolve().parent / "counties"
CENSUS_PATH = Path(__file__).resolve().parent.parent.parent / "docs" / "census_population.csv"
VARIABLES = ["TMAX", "TMIN", "PRCP"]

# Grid epochs: refresh population weights every 4 years
GRID_EPOCH_YEARS = 4

# Map epoch years to their census CSV filenames
CENSUS_FILES = {
    2020: CENSUS_DIR / "census_2020.csv",
    2024: CENSUS_DIR / "census_2024.csv",
}


def grid_epoch_year(year: int) -> int:
    """Round a calendar year down to its nearest available census epoch.

    Uses CENSUS_FILES keys to find the nearest epoch <= year.
    Falls back to the earliest available epoch for years before any file.
    Examples (with files for 2020, 2024):
      2011 -> 2020, 2023 -> 2020, 2024 -> 2024, 2026 -> 2024
    """
    available = sorted(CENSUS_FILES.keys())
    # Find the largest epoch <= year
    candidates = [e for e in available if e <= year]
    if candidates:
        return candidates[-1]
    # Year is before all available epochs -- use earliest
    return available[0]


# -- helpers -----------------------------------------------------------------

def _haversine_matrix(lats1, lons1, lats2, lons2):
    """Pairwise haversine distances (km) between two coordinate arrays."""
    R = 6371.0
    lat1 = np.radians(np.asarray(lats1, dtype=float)[:, None])
    lon1 = np.radians(np.asarray(lons1, dtype=float)[:, None])
    lat2 = np.radians(np.asarray(lats2, dtype=float)[None, :])
    lon2 = np.radians(np.asarray(lons2, dtype=float)[None, :])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    return R * 2 * np.arcsin(np.sqrt(a))


def _load_census(path: str = None, census_year: int = 2024) -> pd.DataFrame:
    """Load census CSV, filter to counties, build 5-digit FIPS.

    Automatically selects the correct census file for the requested epoch
    year from CENSUS_FILES. Falls back to the legacy census_population.csv
    if no epoch-specific file is available.

    Args:
        path: Explicit path to census CSV (overrides auto-selection).
        census_year: Which POPESTIMATE column to use (e.g. 2020, 2024).
    """
    if path is None:
        # Find the census file that contains POPESTIMATE for this year.
        # Try each file (newest first) and pick the first one that has the column.
        for _epoch in sorted(CENSUS_FILES.keys(), reverse=True):
            candidate = CENSUS_FILES[_epoch]
            if candidate.exists():
                sample = pd.read_csv(candidate, encoding="latin-1", nrows=1)
                if f"POPESTIMATE{census_year}" in sample.columns:
                    path = str(candidate)
                    break
        if path is None:
            # Fall back to epoch-based selection
            epoch = grid_epoch_year(census_year)
            path = str(CENSUS_FILES.get(epoch, CENSUS_PATH))
    df = pd.read_csv(path, encoding="latin-1")

    # New census format (census_2020.csv, census_2024.csv):
    #   SUMLEV=050 for counties, FIPS = STATE(2) + COUNTY(3)
    # Legacy format (census_population.csv):
    #   LSAD="County or equivalent", FIPS = STCOU
    if "SUMLEV" in df.columns:
        counties = df[df["SUMLEV"] == 50].copy()
        counties["fips"] = (
            counties["STATE"].astype(str).str.zfill(2)
            + counties["COUNTY"].astype(str).str.zfill(3)
        )
        name_col = "CTYNAME"
    else:
        counties = df[df["LSAD"] == "County or equivalent"].copy()
        counties["fips"] = counties["STCOU"].astype(int).astype(str).str.zfill(5)
        name_col = "NAME"

    pop_col = f"POPESTIMATE{census_year}"
    if pop_col not in counties.columns:
        avail = sorted(
            int(c.replace("POPESTIMATE", ""))
            for c in counties.columns
            if c.startswith("POPESTIMATE")
        )
        nearest = min(avail, key=lambda y: abs(y - census_year))
        print(f"Census year {census_year} not available, using {nearest}")
        pop_col = f"POPESTIMATE{nearest}"

    counties = counties.rename(
        columns={name_col: "county_name", pop_col: "population"}
    )
    return counties[["fips", "county_name", "population"]].reset_index(drop=True)


def _merge_centroids(county_df: pd.DataFrame, gaz_year: int = 2024) -> pd.DataFrame:
    """Merge county population with gazetteer lat/lon centroids."""
    gaz = fetch_gazetteer_counties(year=gaz_year)
    merged = county_df.merge(gaz, left_on="fips", right_on="GEOID", how="left")
    before = len(merged)
    merged = merged.dropna(subset=["latitude"])
    after = len(merged)
    if before != after:
        print(f"Dropped {before - after} counties without centroid match")
    merged["state_fips"] = merged["fips"].str[:2]
    return merged[
        ["fips", "county_name", "population", "latitude", "longitude", "state_fips"]
    ].reset_index(drop=True)


# -- main class --------------------------------------------------------------


class PopulationWeatherGrid:
    """County-level population-weighted temperature and precipitation grid."""

    def __init__(
        self,
        census_path: str = None,
        ncei_token: str = None,
        census_year: int = None,
        max_clusters: int = 15,
        min_coverage: float = 0.8,
        max_stations_per_cluster: int = 2,
        search_radius_km: float = 50.0,
    ):
        self.census_path = census_path
        self.census_year = census_year or grid_epoch_year(date.today().year)
        self.max_clusters = max_clusters
        self.min_coverage = min_coverage
        self.max_stations_per_cluster = max_stations_per_cluster
        self.search_radius_km = search_radius_km

        self._locator = DensityBasedLocator(ncei_token)
        self._api = AgClimateAPI(ncei_token)

        # cached state -- populated by setup()
        self._county_df: Optional[pd.DataFrame] = None
        self._clusters: Optional[List[DensityCluster]] = None
        self._stations: Optional[List[StationMatch]] = None
        self._county_station_map: Optional[pd.DataFrame] = None

        # weather data cache: (start, end) -> pivoted daily DataFrame
        self._daily_cache: dict = {}
        # merged cache: (start, end) -> daily merged with county weights
        self._merged_cache: dict = {}

    # -- setup ---------------------------------------------------------------

    def setup(
        self,
        start_date: date = None,
        end_date: date = None,
    ) -> "PopulationWeatherGrid":
        """
        Load census data, cluster by population, find GHCND stations,
        and build county-to-station mapping. Results are cached on self.

        Args:
            start_date: Used for station data-coverage check (default: 1 year ago)
            end_date: Used for station data-coverage check (default: today)
        """
        if start_date is None:
            start_date = date.today() - timedelta(days=365)
        if end_date is None:
            end_date = date.today()

        # 1. Census + centroids
        print(f"Loading census population data (epoch {self.census_year})...")
        self._county_df = _merge_centroids(
            _load_census(self.census_path, census_year=self.census_year)
        )
        print(f"  {len(self._county_df)} counties with coordinates")

        # 2. Population-density clustering
        print("Clustering counties by population density...")
        self._clusters = self._locator.identify_density_clusters(
            data=self._county_df,
            density_column="population",
            county_column="county_name",
            lat_column="latitude",
            lon_column="longitude",
            fips_column="fips",
            method="weighted_kmeans",
            max_clusters=self.max_clusters,
        )

        # 3. Find stations
        print("Finding optimal GHCND stations...")
        self._stations = self._locator.find_optimal_stations(
            clusters=self._clusters,
            dataset="GHCND",
            variables=VARIABLES,
            start_date=start_date,
            end_date=end_date,
            min_coverage=self.min_coverage,
            max_stations_per_cluster=self.max_stations_per_cluster,
            search_radius_km=self.search_radius_km,
        )
        print(f"  {len(self._stations)} stations selected")

        # 4. Map counties to nearest station
        self._county_station_map = self._build_county_station_map()
        print(f"  Mapped {len(self._county_station_map)} counties to stations")
        return self

    def _build_county_station_map(self) -> pd.DataFrame:
        """Assign each county to its nearest station via haversine."""
        stn_ids = [s.station_id for s in self._stations]
        stn_lats = np.array([s.latitude for s in self._stations])
        stn_lons = np.array([s.longitude for s in self._stations])

        c_lats = self._county_df["latitude"].values
        c_lons = self._county_df["longitude"].values

        dist = _haversine_matrix(c_lats, c_lons, stn_lats, stn_lons)
        nearest_idx = np.argmin(dist, axis=1)

        mapping = self._county_df.copy()
        mapping["station_id"] = [stn_ids[i] for i in nearest_idx]
        mapping["station_dist_km"] = dist[np.arange(len(dist)), nearest_idx]
        return mapping

    # -- data fetch ----------------------------------------------------------

    def _ensure_setup(self):
        if self._county_station_map is None:
            self.setup()

    def _stations_df(self) -> pd.DataFrame:
        """Build the DataFrame that AgClimateAPI._get_ghcnd_data expects."""
        rows = []
        seen = set()
        for s in self._stations:
            if s.station_id in seen:
                continue
            seen.add(s.station_id)
            rows.append(
                {
                    "id": s.station_id,
                    "name": s.station_name,
                    "latitude": s.latitude,
                    "longitude": s.longitude,
                    "elevation": None,
                }
            )
        return pd.DataFrame(rows)

    def clear_cache(self) -> None:
        """Drop all cached weather data, forcing fresh API calls."""
        self._daily_cache.clear()
        self._merged_cache.clear()

    def fetch_data(
        self,
        start_date: date,
        end_date: date,
        variables: List[str] = None,
    ) -> pd.DataFrame:
        """
        Fetch raw GHCND data for all mapped stations.

        Results are cached internally -- repeated calls with the same date
        range return instantly.  Use ``clear_cache()`` to force a refresh.
        """
        self._ensure_setup()
        variables = variables or VARIABLES
        stations_df = self._stations_df()
        return self._api._get_ghcnd_data(stations_df, start_date, end_date, variables)

    # -- pivot / weight ------------------------------------------------------

    @staticmethod
    def _pivot_weather(raw: pd.DataFrame) -> pd.DataFrame:
        """Pivot long-format GHCND data to one row per (date, station)."""
        if raw is None or raw.empty:
            return pd.DataFrame()

        raw = raw.copy()
        raw["date"] = pd.to_datetime(raw["date"])
        pivoted = raw.pivot_table(
            index=["date", "station"],
            columns="datatype",
            values="value",
            aggfunc="first",
        ).reset_index()
        pivoted.columns.name = None

        if "TMAX" in pivoted.columns and "TMIN" in pivoted.columns:
            pivoted["TAVG"] = (pivoted["TMAX"] + pivoted["TMIN"]) / 2

        pivoted = pivoted.rename(columns={"station": "station_id"})
        return pivoted

    def _get_daily_pivoted(self, start_date: date, end_date: date) -> pd.DataFrame:
        """Return pivoted daily data, hitting the API only on cache miss."""
        key = (start_date, end_date)
        if key not in self._daily_cache:
            raw = self.fetch_data(start_date, end_date)
            self._daily_cache[key] = self._pivot_weather(raw)
        return self._daily_cache[key]

    def _get_merged(self, start_date: date, end_date: date) -> pd.DataFrame:
        """Return daily data merged with county weights, cached."""
        key = (start_date, end_date)
        if key not in self._merged_cache:
            daily = self._get_daily_pivoted(start_date, end_date)
            if daily.empty:
                self._merged_cache[key] = daily
            else:
                csm = self._county_station_map[
                    ["fips", "station_id", "population", "state_fips"]
                ]
                self._merged_cache[key] = daily.merge(csm, on="station_id", how="inner")
        return self._merged_cache[key]

    def get_weighted_daily(
        self,
        start_date: date,
        end_date: date,
        level: str = "national",
    ) -> pd.DataFrame:
        """
        Population-weighted daily weather.

        Args:
            start_date: Start date
            end_date: End date
            level: 'national', 'state', or 'county'

        Returns:
            DataFrame indexed by date with weighted weather columns.
        """
        merged = self._get_merged(start_date, end_date)
        if merged.empty:
            return merged

        value_cols = [c for c in ["TAVG", "TMAX", "TMIN", "PRCP"] if c in merged.columns]

        if level == "county":
            return (
                merged.set_index("date")[
                    ["fips", "state_fips", "population"] + value_cols
                ]
                .sort_index()
            )

        if level == "state":
            group_key = ["date", "state_fips"]
        else:
            group_key = ["date"]

        df = merged.dropna(subset=value_cols, how="all").copy()

        df["weight"] = df.groupby(group_key)["population"].transform(
            lambda x: x / x.sum()
        )

        for col in value_cols:
            df[f"wtd_{col}"] = df[col] * df["weight"]

        wtd_cols = [f"wtd_{c}" for c in value_cols]

        if level == "state":
            result = df.groupby(group_key)[wtd_cols].sum().reset_index()
            result = result.set_index("date").sort_index()
        else:
            result = df.groupby("date")[wtd_cols].sum()
            result = result.sort_index()

        return result

    def get_weighted_monthly(
        self,
        start_date: date,
        end_date: date,
        level: str = "national",
    ) -> pd.DataFrame:
        """
        Population-weighted monthly weather.

        Temperature columns are averaged; PRCP is summed (monthly total).
        """
        daily = self.get_weighted_daily(start_date, end_date, level=level)
        if daily.empty:
            return daily

        temp_cols = [c for c in daily.columns if "TAVG" in c or "TMAX" in c or "TMIN" in c]
        prcp_cols = [c for c in daily.columns if "PRCP" in c]

        if level == "state":
            groups = daily.groupby("state_fips")
            parts = []
            for sf, grp in groups:
                temps = grp[temp_cols].resample("MS").mean() if temp_cols else pd.DataFrame()
                prcp = grp[prcp_cols].resample("MS").sum() if prcp_cols else pd.DataFrame()
                monthly = pd.concat([temps, prcp], axis=1)
                monthly["state_fips"] = sf
                parts.append(monthly)
            return pd.concat(parts).sort_index()
        else:
            temps = daily[temp_cols].resample("MS").mean() if temp_cols else pd.DataFrame()
            prcp = daily[prcp_cols].resample("MS").sum() if prcp_cols else pd.DataFrame()
            return pd.concat([temps, prcp], axis=1).sort_index()

    # -- persistence ---------------------------------------------------------

    def save_config(self, path: str = None) -> None:
        """Save station-to-county mapping so setup() can be skipped later.

        If *path* is omitted, a default name including the census epoch is used.
        """
        if path is None:
            path = f"pop_weather_config_{self.census_year}.json"
        data = {
            "census_year": self.census_year,
            "stations": [
                {
                    "station_id": s.station_id,
                    "station_name": s.station_name,
                    "latitude": s.latitude,
                    "longitude": s.longitude,
                    "cluster_id": s.cluster_id,
                    "distance_km": s.distance_km,
                    "data_coverage": s.data_coverage,
                    "match_score": s.match_score,
                }
                for s in self._stations
            ],
            "county_station_map": self._county_station_map.to_dict(orient="records"),
        }
        Path(path).write_text(json.dumps(data, indent=2))
        print(f"Saved config to {path} (census_year={self.census_year})")

    @classmethod
    def from_config(cls, path: str, ncei_token: str = None) -> "PopulationWeatherGrid":
        """Restore from a previously saved config, skipping setup()."""
        data = json.loads(Path(path).read_text())
        census_year = data.get("census_year", 2024)
        grid = cls(ncei_token=ncei_token, census_year=census_year)
        grid._stations = [
            StationMatch(**s) for s in data["stations"]
        ]
        grid._county_station_map = pd.DataFrame(data["county_station_map"])
        return grid

    @property
    def needs_refresh(self) -> bool:
        """True if current calendar year has moved to a new 4-year epoch."""
        return grid_epoch_year(date.today().year) != self.census_year
