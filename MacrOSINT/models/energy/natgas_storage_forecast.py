"""
Natural Gas Storage Forecaster.

Uses population-weighted Heating/Cooling Degree Days from PopulationWeatherGrid
combined with EIA weekly underground storage data to forecast weekly storage
changes via SARIMAX regression.

References:
    - Gas-Weighted Degree Days (GWDD) methodology
    - SARIMAX(1,1,1)(1,1,1,52) with exogenous weather + price variables
    - Piecewise / spline HDD for non-linear cold-weather demand response
    - ConsensusForecast: street estimate proxy via seasonal range positioning
"""
import warnings
from datetime import date, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX

from MacrOSINT.data.sources.eia.api_tools import NatGasHelper
from MacrOSINT.models.weather.population_weather import (
    CENSUS_FILES,
    PopulationWeatherGrid,
    grid_epoch_year,
)

# Optional sklearn for spline basis and residual correction
try:
    from sklearn.preprocessing import SplineTransformer
    from sklearn.ensemble import GradientBoostingRegressor
    _SKLEARN_AVAILABLE = True
except ImportError:
    _SKLEARN_AVAILABLE = False

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

BASE_TEMP_F = 65.0
BASE_TEMP_C = (BASE_TEMP_F - 32) * 5 / 9  # ~18.33 C
MILD_HDD_CUTOFF = 15.0                      # degrees F below base


# ---------------------------------------------------------------------------
# Degree-day calculations
# ---------------------------------------------------------------------------

def compute_degree_days(
    daily_temps: pd.DataFrame,
    base_temp: float = BASE_TEMP_C,
    temp_col: str = "wtd_TAVG",
    celsius: bool = True,
) -> pd.DataFrame:
    """
    Compute HDD, CDD, and piecewise HDD from population-weighted daily temps.

    Returns DataFrame with columns: HDD, CDD, HDD_mild, HDD_extreme
    """
    if temp_col not in daily_temps.columns:
        raise KeyError(f"Column '{temp_col}' not found. Available: {list(daily_temps.columns)}")

    tavg = daily_temps[temp_col].copy()

    cutoff_c = MILD_HDD_CUTOFF * 5 / 9 if celsius else MILD_HDD_CUTOFF
    hdd = (base_temp - tavg).clip(lower=0)
    cdd = (tavg - base_temp).clip(lower=0)

    return pd.DataFrame({
        "HDD":         hdd,
        "CDD":         cdd,
        "HDD_mild":    hdd.clip(upper=cutoff_c),
        "HDD_extreme": (hdd - cutoff_c).clip(lower=0),
    }, index=daily_temps.index)


def resample_weekly(degree_days: pd.DataFrame, agg: str = "sum") -> pd.DataFrame:
    """Resample daily degree days to weekly (Friday-ending to match EIA)."""
    return degree_days.resample("W-FRI").sum() if agg == "sum" else degree_days.resample("W-FRI").mean()


# ---------------------------------------------------------------------------
# Spline HDD basis
# ---------------------------------------------------------------------------

def compute_spline_hdd_basis(
    hdd: pd.Series,
    n_knots: int = 4,
    transformer=None,
) -> Tuple[pd.DataFrame, object]:
    """
    Natural cubic spline basis expansion for HDD.

    Replaces piecewise HDD_mild/HDD_extreme with a smooth spline that
    captures the convex demand response at extreme cold temperatures.

    Args:
        hdd: Weekly HDD series (non-negative).
        n_knots: Interior knots (placed at HDD quantiles).
        transformer: Pre-fitted SplineTransformer; if None, fits on hdd.

    Returns:
        (basis_df, transformer) tuple.
    """
    if not _SKLEARN_AVAILABLE:
        raise ImportError("scikit-learn required for spline HDD. pip install scikit-learn")

    values = hdd.values.reshape(-1, 1)
    if transformer is None:
        transformer = SplineTransformer(
            n_knots=n_knots,
            degree=3,
            knots="quantile",
            extrapolation="linear",
            include_bias=False,
        )
        transformer.fit(values)

    basis = transformer.transform(values)
    cols = [f"HDD_sp{i}" for i in range(basis.shape[1])]
    return pd.DataFrame(basis, index=hdd.index, columns=cols), transformer


# ---------------------------------------------------------------------------
# Fourier seasonal terms
# ---------------------------------------------------------------------------

def add_fourier_terms(
    df: pd.DataFrame,
    period: int = 52,
    n_harmonics: int = 2,
) -> pd.DataFrame:
    """
    Add sin/cos Fourier terms for annual and semi-annual seasonality.

    Args:
        df: DataFrame with DatetimeIndex.
        period: Seasonal period in weeks.
        n_harmonics: Number of harmonics (1=annual, 2=annual+semi-annual).

    Returns:
        Copy of df with added Fourier columns.
    """
    result = df.copy()
    week = pd.Series(
        df.index.isocalendar().week.astype(float), index=df.index
    )
    for h in range(1, n_harmonics + 1):
        result[f"sin_{h}"] = np.sin(2 * np.pi * h * week / period)
        result[f"cos_{h}"] = np.cos(2 * np.pi * h * week / period)
    return result


# ---------------------------------------------------------------------------
# Storage data helpers
# ---------------------------------------------------------------------------

def fetch_storage_data(
    ng_helper: NatGasHelper,
    start: Optional[str] = None,
    end: Optional[str] = None,
) -> pd.DataFrame:
    """
    Fetch EIA weekly underground natural gas storage and compute
    week-over-week change.

    Returns DataFrame with columns: storage_level, storage_change
    """
    raw = ng_helper.get_underground_storage(start=start, end=end)
    if raw is None or raw.empty:
        raise ValueError("No storage data returned from EIA API")

    if "period" in raw.columns:
        raw["period"] = pd.to_datetime(raw["period"])
        raw = raw.set_index("period").sort_index()

    raw = raw.drop(columns=["units"], errors="ignore")

    if "R48" in raw.columns:
        storage = raw["R48"].astype(float)
    else:
        numeric_cols = raw.select_dtypes(include="number").columns
        storage = raw[numeric_cols].sum(axis=1)

    df = pd.DataFrame({"storage_level": storage}, index=raw.index)
    df.index.name = "date"
    df["storage_change"] = df["storage_level"].diff()
    return df.dropna(subset=["storage_change"])


def fetch_spot_prices(
    ng_helper: NatGasHelper,
    start: Optional[str] = None,
    end: Optional[str] = None,
    chunk_years: int = 3,
) -> pd.Series:
    """
    Fetch Henry Hub spot prices from EIA and resample to weekly Friday averages.

    Fetches in yearly chunks to avoid EIA server 504 timeouts on large
    daily date ranges (15+ years of daily data exceeds the API's capacity).
    """
    start_ts = pd.Timestamp(start or "2000-01")
    end_ts = pd.Timestamp(end or date.today().strftime("%Y-%m"))

    chunks = []
    yr = start_ts.year
    while yr <= end_ts.year:
        chunk_end_yr = min(yr + chunk_years - 1, end_ts.year)
        chunk_start = f"{yr}-01"
        chunk_end = f"{chunk_end_yr}-12"
        try:
            raw = ng_helper.execute_request("spot_prices", start=chunk_start, end=chunk_end)
            if raw is not None and not raw.empty:
                chunks.append(raw)
        except Exception as e:
            warnings.warn(f"Price fetch failed for {chunk_start}-{chunk_end}: {e}")
        yr += chunk_years

    if not chunks:
        warnings.warn("No spot price data returned; model will run without price feature.")
        return pd.Series(dtype=float)

    raw = pd.concat(chunks)
    if "period" in raw.columns:
        raw["period"] = pd.to_datetime(raw["period"])
        raw = raw.set_index("period").sort_index()
    raw = raw[~raw.index.duplicated(keep="last")]
    raw = raw.drop(columns=["units"], errors="ignore")
    numeric_cols = raw.select_dtypes(include="number").columns
    if len(numeric_cols) == 0:
        warnings.warn("No numeric price columns found.")
        return pd.Series(dtype=float)

    weekly = raw[numeric_cols[0]].astype(float).resample("W-FRI").mean()
    weekly.name = "gas_price"
    return weekly


def fetch_lng_exports(
    ng_helper: NatGasHelper,
    start: Optional[str] = None,
    end: Optional[str] = None,
) -> pd.Series:
    """
    Fetch monthly LNG export volumes (EIA /move/poe2, process=ENG) and
    forward-fill to a weekly (Friday-ending) series.

    Sums across all border-crossing duoareas (NUS-NCA, NUS-NMX, NUS-Z00) to
    produce total US LNG exports. The NUS-Z00 bucket dominates, capturing
    feedgas to Sabine Pass, Corpus Christi, Freeport, Cameron, and Cove Point.

    Monthly values are forward-filled to weekly so each week in a given month
    carries that month's export rate as an exogenous demand signal.

    Returns:
        Weekly Series named 'lng_exports', or empty Series on failure.
    """
    try:
        raw = ng_helper.get_lng_exports(start=start, end=end)
    except Exception as e:
        warnings.warn(f"LNG export fetch failed ({e}); model will run without LNG feature.")
        return pd.Series(dtype=float, name="lng_exports")

    if raw is None or raw.empty:
        warnings.warn("No LNG export data returned; model will run without LNG feature.")
        return pd.Series(dtype=float, name="lng_exports")

    if "period" in raw.columns:
        raw["period"] = pd.to_datetime(raw["period"])
        raw = raw.set_index("period").sort_index()

    raw = raw.drop(columns=["units"], errors="ignore")
    numeric_cols = raw.select_dtypes(include="number").columns
    if len(numeric_cols) == 0:
        return pd.Series(dtype=float, name="lng_exports")

    # Sum all destination columns → total US LNG exports (Bcf/month)
    monthly_total = raw[numeric_cols].sum(axis=1).astype(float)
    monthly_total.name = "lng_exports"
    monthly_total.index = pd.to_datetime(monthly_total.index).to_period("M").to_timestamp("M")

    # Resample monthly → weekly (forward-fill so each week carries month's rate)
    weekly_idx = pd.date_range(
        start=monthly_total.index.min(),
        end=monthly_total.index.max() + pd.DateOffset(months=1),
        freq="W-FRI",
    )
    weekly = monthly_total.reindex(weekly_idx, method="ffill")
    weekly.name = "lng_exports"
    return weekly


# ---------------------------------------------------------------------------
# Consensus (street estimate) forecast
# ---------------------------------------------------------------------------

class ConsensusForecast:
    """
    Street estimate proxy for EIA weekly natural gas storage.

    Estimates market consensus by blending 5-year rolling seasonal statistics
    for the same week-of-year. Empirical analysis of M0-M1 spread reactions
    shows the market prices storage relative to the top of the seasonal range
    (5yr max) more than any model-based forecast.

    Default weights (optimized against M0-M1 spread reaction, 2014-2025):
        5yr Max    : 0.62
        5yr Mean   : 0.28
        5yr Median : 0.10

    Example:
        consensus = ConsensusForecast()
        consensus.fit(storage_df['storage_change'])
        result = consensus.transform()
        # result has: actual, consensus_est, surprise, sea_mean, sea_max, ...
    """

    DEFAULT_WEIGHTS: Dict[str, float] = {
        "sea_max":  0.62,
        "sea_mean": 0.28,
        "sea_med":  0.10,
        "sea_min":  0.00,
    }

    def __init__(
        self,
        lookback_years: int = 5,
        min_obs: int = 3,
        weights: Optional[Dict[str, float]] = None,
    ):
        """
        Args:
            lookback_years: Rolling window for seasonal stats.
            min_obs: Minimum same-week-of-year observations required.
            weights: Override default component weights. Keys: sea_mean,
                     sea_med, sea_min, sea_max. Auto-normalized to sum to 1.
        """
        self.lookback_years = lookback_years
        self.min_obs = min_obs
        raw_w = weights or self.DEFAULT_WEIGHTS
        total = sum(raw_w.values()) or 1.0
        self.weights = {k: v / total for k, v in raw_w.items()}
        self._history: Optional[pd.Series] = None

    def fit(self, storage_series: pd.Series) -> "ConsensusForecast":
        """
        Store historical weekly storage changes.

        Args:
            storage_series: Series of actual weekly storage changes,
                            indexed by week-ending date.
        """
        self._history = storage_series.copy().sort_index()
        self._history.name = "actual"
        return self

    def _seasonal_stats(self, target_date: pd.Timestamp, week: int) -> dict:
        """Compute 5yr rolling seasonal stats for a given week-of-year."""
        cutoff = target_date - pd.DateOffset(years=self.lookback_years)
        hist = self._history[
            (self._history.index < target_date) &
            (self._history.index >= cutoff)
        ]
        wk_hist = hist[hist.index.isocalendar().week.astype(int) == week]
        if len(wk_hist) < self.min_obs:
            return {}
        return {
            "sea_mean": float(wk_hist.mean()),
            "sea_med":  float(wk_hist.median()),
            "sea_min":  float(wk_hist.min()),
            "sea_max":  float(wk_hist.max()),
            "n_obs":    len(wk_hist),
        }

    def estimate(self, target_date) -> Optional[float]:
        """
        Compute street estimate for a single week-ending date.

        Returns estimated storage change (Bcf) or None if insufficient history.
        """
        if self._history is None:
            raise RuntimeError("Call fit() before estimate().")
        ts = pd.Timestamp(target_date)
        week = int(ts.isocalendar().week)
        stats = self._seasonal_stats(ts, week)
        if not stats:
            return None
        return sum(
            self.weights.get(k, 0.0) * stats[k]
            for k in ("sea_mean", "sea_med", "sea_min", "sea_max")
            if k in stats
        )

    def transform(
        self,
        storage_series: Optional[pd.Series] = None,
    ) -> pd.DataFrame:
        """
        Build time series of consensus estimates and surprises.

        Args:
            storage_series: Series to compute estimates for. Defaults to
                            full fitted history.

        Returns:
            DataFrame with columns: actual, consensus_est, surprise,
            sea_mean, sea_med, sea_min, sea_max, n_obs.
        """
        if self._history is None:
            raise RuntimeError("Call fit() before transform().")

        series = storage_series if storage_series is not None else self._history
        rows = []
        for dt, actual in series.items():
            ts = pd.Timestamp(dt)
            week = int(ts.isocalendar().week)
            stats = self._seasonal_stats(ts, week)
            if not stats:
                continue
            est = sum(
                self.weights.get(k, 0.0) * stats[k]
                for k in ("sea_mean", "sea_med", "sea_min", "sea_max")
                if k in stats
            )
            rows.append({
                "date":          dt,
                "actual":        actual,
                "consensus_est": est,
                "surprise":      actual - est,
                "sea_mean":      stats["sea_mean"],
                "sea_med":       stats["sea_med"],
                "sea_min":       stats["sea_min"],
                "sea_max":       stats["sea_max"],
                "n_obs":         stats["n_obs"],
            })

        result = pd.DataFrame(rows).set_index("date")
        result.index = pd.to_datetime(result.index)
        return result.sort_index()

    def update(self, new_actual: pd.Series) -> "ConsensusForecast":
        """Append new observations to history."""
        if self._history is None:
            self._history = new_actual.copy().sort_index()
        else:
            combined = pd.concat([self._history, new_actual]).sort_index()
            self._history = combined[~combined.index.duplicated(keep="last")]
        return self


# ---------------------------------------------------------------------------
# Forecaster
# ---------------------------------------------------------------------------

class NatGasStorageForecaster:
    """
    Weekly natural gas storage change forecaster using population-weighted
    degree days and SARIMAX regression.

    Feature improvements over baseline:
      - Spline HDD basis (replaces piecewise mild/extreme)
      - Fourier seasonal terms (annual + semi-annual harmonics)
      - Storage deficit vs 5yr seasonal norm (mean-reversion regime)
      - Optional gradient-boosted residual correction layer

    Usage:
        forecaster = NatGasStorageForecaster()
        features = forecaster.build_features('2011-01', '2026-03',
                                              daily_weather=weather_df)
        forecaster.fit(data=features[features.index < cutoff])
        forecast = forecaster.forecast(steps=4, future_exog=future_features)
    """

    def __init__(
        self,
        weather_grid: Optional[PopulationWeatherGrid] = None,
        ng_helper: Optional[NatGasHelper] = None,
        ncei_token: Optional[str] = None,
        config_dir: Optional[str] = None,
        base_temp_c: float = BASE_TEMP_C,
        order: Tuple[int, int, int] = (1, 1, 1),
        seasonal_order: Tuple[int, int, int, int] = (1, 0, 1, 52),
        use_price: bool = True,
        use_piecewise_hdd: bool = True,   # kept for backward compat; overridden by use_spline_hdd
        use_spline_hdd: bool = True,
        n_spline_knots: int = 4,
        use_fourier: bool = True,
        n_fourier_harmonics: int = 2,
        use_storage_norm: bool = True,
        use_lng_exports: bool = True,
        use_residual_correction: bool = False,
    ):
        self.ng_helper = ng_helper or NatGasHelper()
        self.ncei_token = ncei_token
        self.config_dir = Path(config_dir) if config_dir else None
        self.base_temp_c = base_temp_c
        self.order = order
        self.seasonal_order = seasonal_order
        self.use_price = use_price
        self.use_piecewise_hdd = use_piecewise_hdd
        self.use_spline_hdd = use_spline_hdd and _SKLEARN_AVAILABLE
        self.n_spline_knots = n_spline_knots
        self.use_fourier = use_fourier
        self.n_fourier_harmonics = n_fourier_harmonics
        self.use_storage_norm = use_storage_norm
        self.use_lng_exports = use_lng_exports
        self.use_residual_correction = use_residual_correction

        if use_spline_hdd and not _SKLEARN_AVAILABLE:
            warnings.warn("scikit-learn not found; falling back to piecewise HDD.")

        # Grid registry
        self._grids: dict = {}
        if weather_grid is not None:
            self._grids[weather_grid.census_year] = weather_grid
        self.weather_grid = weather_grid

        # Fitted model
        self._model = None
        self._results = None
        self._training_data: Optional[pd.DataFrame] = None
        self._exog_cols: list = []
        self._spline_transformer = None
        self._residual_model = None

    # -- setup ---------------------------------------------------------------

    def _config_path_for_epoch(self, epoch_year: int) -> Optional[Path]:
        if self.config_dir is None:
            return None
        for p in [
            self.config_dir / f"pop_weather_config_{epoch_year}.json",
            self.config_dir / "pop_weather_config.json",
        ]:
            if p.exists():
                return p
        return None

    def get_grid(self, epoch_year: int) -> PopulationWeatherGrid:
        if epoch_year not in self._grids:
            config_path = self._config_path_for_epoch(epoch_year)
            if config_path:
                grid = PopulationWeatherGrid.from_config(
                    str(config_path), ncei_token=self.ncei_token
                )
            else:
                grid = PopulationWeatherGrid(
                    ncei_token=self.ncei_token, census_year=epoch_year
                )
                grid.setup()
                if self.config_dir:
                    self.config_dir.mkdir(parents=True, exist_ok=True)
                    grid.save_config(
                        str(self.config_dir / f"pop_weather_config_{epoch_year}.json")
                    )
            self._grids[epoch_year] = grid
        return self._grids[epoch_year]

    def setup(self, weather_config_path: Optional[str] = None) -> "NatGasStorageForecaster":
        if weather_config_path:
            grid = PopulationWeatherGrid.from_config(
                weather_config_path, ncei_token=self.ncei_token
            )
            self._grids[grid.census_year] = grid
            self.weather_grid = grid
        elif self.weather_grid is None:
            epoch = grid_epoch_year(date.today().year)
            self.weather_grid = self.get_grid(epoch)
        else:
            self.weather_grid.setup()
        return self

    # -- data assembly -------------------------------------------------------

    def _fetch_weather_by_epoch(self, start_dt: date, end_dt: date) -> pd.DataFrame:
        start_epoch = grid_epoch_year(start_dt.year)
        end_epoch = grid_epoch_year(end_dt.year)

        if start_epoch == end_epoch:
            return self.get_grid(start_epoch).get_weighted_daily(
                start_dt, end_dt, level="national"
            )

        epochs = sorted(CENSUS_FILES.keys())
        relevant = [e for e in epochs if e <= end_epoch] or [epochs[0]]

        frames = []
        for i, epoch in enumerate(relevant):
            if epoch < start_epoch:
                continue
            seg_start = max(start_dt, date(epoch, 1, 1))
            seg_end = (
                min(end_dt, date(relevant[i + 1] - 1, 12, 31))
                if i + 1 < len(relevant)
                else end_dt
            )
            chunk = self.get_grid(epoch).get_weighted_daily(
                seg_start, seg_end, level="national"
            )
            if not chunk.empty:
                frames.append(chunk)

        if not frames:
            return pd.DataFrame()
        daily = pd.concat(frames).sort_index()
        return daily[~daily.index.duplicated(keep="first")]

    def _add_storage_deficit(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add storage_deficit = storage_level minus 5yr rolling seasonal mean
        for the same week-of-year.  Captures mean-reversion regime.
        """
        df = df.copy()
        weeks = df.index.isocalendar().week.astype(int)
        deficits = []
        for i, (idx, row) in enumerate(df.iterrows()):
            wk = int(weeks.iloc[i])
            cutoff = idx - pd.DateOffset(years=5)
            hist = df[(df.index < idx) & (df.index >= cutoff)]
            hist_wk = hist[hist.index.isocalendar().week.astype(int) == wk]
            if len(hist_wk) >= 3:
                deficits.append(row["storage_level"] - hist_wk["storage_level"].mean())
            else:
                deficits.append(np.nan)
        df["storage_deficit"] = deficits
        return df

    def build_features(
        self,
        start: str,
        end: str,
        daily_weather: Optional[pd.DataFrame] = None,
        storage_data: Optional[pd.DataFrame] = None,
        price_data: Optional[pd.Series] = None,
        lng_data: Optional[pd.Series] = None,
    ) -> pd.DataFrame:
        """
        Assemble the weekly feature matrix.

        Includes: storage change (target), population-weighted HDD/CDD,
        spline HDD basis, Fourier seasonal terms, storage deficit vs norm,
        LNG exports, and optionally spot price.

        Pre-fetched data may be passed to avoid redundant API calls:
            storage_data: output of fetch_storage_data()
            price_data:   output of fetch_spot_prices()
            lng_data:     output of fetch_lng_exports()
        """
        start_dt = pd.Timestamp(start).date()
        end_dt = pd.Timestamp(end).date()

        # 1. Weather
        daily = daily_weather if daily_weather is not None else \
            self._fetch_weather_by_epoch(start_dt, end_dt)
        if daily.empty:
            raise ValueError("No weather data for the requested period")

        # 2. Degree days
        dd = compute_degree_days(daily, base_temp=self.base_temp_c)
        weekly_dd = resample_weekly(dd)

        # 3. Spline HDD basis (fit on first call, reuse thereafter)
        if self.use_spline_hdd:
            spline_df, self._spline_transformer = compute_spline_hdd_basis(
                weekly_dd["HDD"],
                n_knots=self.n_spline_knots,
                transformer=self._spline_transformer,
            )
            weekly_dd = pd.concat([weekly_dd, spline_df], axis=1)

        # 4. Storage
        storage = storage_data if storage_data is not None else \
            fetch_storage_data(self.ng_helper, start=start, end=end)

        # 5. Spot prices
        if price_data is not None:
            prices = price_data
        elif self.use_price:
            prices = fetch_spot_prices(self.ng_helper, start=start, end=end)
        else:
            prices = pd.Series(dtype=float)

        # 6. Merge
        merged = storage.join(weekly_dd, how="inner")
        if not prices.empty:
            merged = merged.join(prices, how="left")
            merged["gas_price"] = merged["gas_price"].ffill()

        # 6b. LNG exports (monthly forward-filled to weekly)
        if self.use_lng_exports:
            if lng_data is not None:
                lng = lng_data
            else:
                lng = fetch_lng_exports(self.ng_helper, start=start, end=end)
            if not lng.empty:
                merged = merged.join(lng, how="left")
                merged["lng_exports"] = merged["lng_exports"].ffill()

        # 7. Storage deficit vs 5yr norm
        if self.use_storage_norm:
            merged = self._add_storage_deficit(merged)

        # 8. Fourier seasonal terms
        if self.use_fourier:
            merged = add_fourier_terms(
                merged, period=52, n_harmonics=self.n_fourier_harmonics
            )

        return merged.dropna()

    def _select_exog_cols(self, df: pd.DataFrame) -> List[str]:
        """Select exogenous columns based on current configuration."""
        cols = []

        # HDD: spline basis takes priority over piecewise
        if self.use_spline_hdd:
            spline_cols = sorted(c for c in df.columns if c.startswith("HDD_sp"))
            if spline_cols:
                cols.extend(spline_cols)
            else:
                cols.extend(["HDD_mild", "HDD_extreme"] if self.use_piecewise_hdd else ["HDD"])
        else:
            cols.extend(["HDD_mild", "HDD_extreme"] if self.use_piecewise_hdd else ["HDD"])

        cols.append("CDD")

        if self.use_price and "gas_price" in df.columns:
            cols.append("gas_price")

        if self.use_storage_norm and "storage_deficit" in df.columns:
            cols.append("storage_deficit")

        if self.use_lng_exports and "lng_exports" in df.columns:
            cols.append("lng_exports")

        if self.use_fourier:
            for h in range(1, self.n_fourier_harmonics + 1):
                cols += [f"sin_{h}", f"cos_{h}"]

        return [c for c in cols if c in df.columns]

    # -- model ---------------------------------------------------------------

    def fit(
        self,
        start: str = "2020-01",
        end: Optional[str] = None,
        data: Optional[pd.DataFrame] = None,
    ) -> "NatGasStorageForecaster":
        """
        Fit the SARIMAX model on historical data.

        Args:
            start: Training period start (used only if data is None).
            end: Training period end (default: today).
            data: Pre-built feature DataFrame from build_features().
        """
        if end is None:
            end = date.today().strftime("%Y-%m")
        if data is None:
            data = self.build_features(start, end)

        self._training_data = data.copy()
        self._exog_cols = self._select_exog_cols(data)

        endog = data["storage_change"]
        exog = data[self._exog_cols] if self._exog_cols else None

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self._model = SARIMAX(
                endog,
                exog=exog,
                order=self.order,
                seasonal_order=self.seasonal_order,
                enforce_stationarity=False,
                enforce_invertibility=False,
            )
            self._results = self._model.fit(disp=False, maxiter=200)

        return self

    def fit_residual_correction(
        self,
        data: Optional[pd.DataFrame] = None,
        n_estimators: int = 100,
        max_depth: int = 3,
        learning_rate: float = 0.05,
    ) -> "NatGasStorageForecaster":
        """
        Fit a gradient-boosted residual correction layer.

        Trains a GradientBoostingRegressor on SARIMAX in-sample residuals
        using the same exogenous features. The correction is applied
        additively during forecast() when use_residual_correction=True.

        Args:
            data: Feature DataFrame (defaults to stored training data).
            n_estimators: Boosting rounds.
            max_depth: Tree depth.
            learning_rate: Shrinkage.
        """
        if not _SKLEARN_AVAILABLE:
            warnings.warn("scikit-learn required for residual correction.")
            return self
        if self._results is None:
            raise RuntimeError("Call fit() before fit_residual_correction().")

        data = data if data is not None else self._training_data
        insample = self.insample_fit()
        residuals = insample["residual"]

        exog_data = data.loc[residuals.index]
        X = exog_data[[c for c in self._exog_cols if c in exog_data.columns]].values
        y = residuals.values

        self._residual_model = GradientBoostingRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            subsample=0.8,
            random_state=42,
        )
        self._residual_model.fit(X, y)

        r = np.corrcoef(y, self._residual_model.predict(X))[0, 1]
        print(f"Residual correction: in-sample residual r={r:.3f}")
        return self

    def summary(self):
        if self._results is None:
            raise RuntimeError("Model not fitted. Call fit() first.")
        return self._results.summary()

    def forecast(
        self,
        steps: int = 4,
        future_exog: Optional[pd.DataFrame] = None,
    ) -> pd.DataFrame:
        """
        Forecast weekly storage changes.

        Args:
            steps: Weeks ahead.
            future_exog: DataFrame with exogenous variables for the forecast
                         horizon. Should contain the same raw feature columns
                         returned by build_features() (spline/Fourier columns
                         included if build_features() was used to construct it).

        Returns:
            DataFrame with columns: forecast, lower_ci, upper_ci
        """
        if self._results is None:
            raise RuntimeError("Model not fitted. Call fit() first.")
        if self._exog_cols and future_exog is None:
            raise ValueError(
                f"Model trained with exog {self._exog_cols}. Provide future_exog."
            )

        exog = future_exog[self._exog_cols] if future_exog is not None else None
        fc = self._results.get_forecast(steps=steps, exog=exog)
        pred = fc.predicted_mean
        ci = fc.conf_int(alpha=0.05)

        result = pd.DataFrame({
            "forecast":  pred.values,
            "lower_ci":  ci.iloc[:, 0].values,
            "upper_ci":  ci.iloc[:, 1].values,
        }, index=pred.index)

        # Residual correction
        if (
            self.use_residual_correction
            and self._residual_model is not None
            and future_exog is not None
        ):
            feat_cols = [c for c in self._exog_cols if c in future_exog.columns]
            correction = self._residual_model.predict(future_exog[feat_cols].values)
            result["forecast"]  += correction
            result["lower_ci"]  += correction
            result["upper_ci"]  += correction

        return result

    def forecast_storage_levels(
        self,
        steps: int = 4,
        future_exog: Optional[pd.DataFrame] = None,
    ) -> pd.DataFrame:
        """Forecast absolute storage levels by accumulating predicted changes."""
        changes = self.forecast(steps=steps, future_exog=future_exog)
        last_level = self._training_data["storage_level"].iloc[-1]
        changes["level_forecast"] = last_level + changes["forecast"].cumsum()
        changes["level_lower"]    = last_level + changes["lower_ci"].cumsum()
        changes["level_upper"]    = last_level + changes["upper_ci"].cumsum()
        return changes

    # -- diagnostics ---------------------------------------------------------

    def insample_fit(self) -> pd.DataFrame:
        if self._results is None:
            raise RuntimeError("Model not fitted. Call fit() first.")
        fitted = self._results.fittedvalues
        actual = self._training_data["storage_change"]
        return pd.DataFrame({
            "actual":   actual,
            "fitted":   fitted,
            "residual": actual - fitted,
        })

    @property
    def aic(self) -> float:
        if self._results is None:
            raise RuntimeError("Model not fitted.")
        return self._results.aic

    @property
    def bic(self) -> float:
        if self._results is None:
            raise RuntimeError("Model not fitted.")
        return self._results.bic

    # -- HDF5 persistence ----------------------------------------------------

    @staticmethod
    def save_weather_hdf(
        daily_weather: pd.DataFrame,
        hdf_path: str = r"F:\Data\weather.hdf",
        key: str = "weather/ng/storage_weather_data",
    ) -> None:
        path = Path(hdf_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        daily_weather.to_hdf(str(path), key=key, mode="a", format="table")
        print(f"Saved weather to {path} [key={key}] ({len(daily_weather)} rows)")

    @staticmethod
    def load_weather_hdf(
        hdf_path: str = r"F:\Data\weather.hdf",
        key: str = "weather/ng/storage_weather_data",
    ) -> Optional[pd.DataFrame]:
        path = Path(hdf_path)
        if not path.exists():
            return None
        try:
            df = pd.read_hdf(str(path), key=key)
            df.index = pd.to_datetime(df.index)
            return df
        except KeyError:
            return None

    @staticmethod
    def _hdf_load(hdf_path: str, key: str) -> Optional[pd.DataFrame]:
        path = Path(hdf_path)
        if not path.exists():
            return None
        try:
            df = pd.read_hdf(str(path), key=key)
            df.index = pd.to_datetime(df.index)
            return df
        except KeyError:
            return None

    @staticmethod
    def _hdf_save(df: pd.DataFrame, hdf_path: str, key: str, label: str = "") -> None:
        path = Path(hdf_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        df.to_hdf(str(path), key=key, mode="a", format="table")
        if label:
            print(f"Cached {label} to {path} [{key}] ({len(df)} rows)")

    @staticmethod
    def load_eia_cache(
        hdf_path: str = r"F:\Data\ng_eia_cache.hdf",
    ) -> dict:
        """
        Load cached EIA data (storage, prices, LNG exports) from HDF5.
        Returns dict with keys 'storage', 'prices', 'lng_exports' (None if missing).
        """
        result = {}
        for key, attr in [
            ("ng/storage", "storage"),
            ("ng/prices", "prices"),
            ("ng/lng_exports", "lng_exports"),
        ]:
            result[attr] = NatGasStorageForecaster._hdf_load(hdf_path, key)
        return result

    @staticmethod
    def save_eia_cache(
        storage: Optional[pd.DataFrame] = None,
        prices: Optional[pd.Series] = None,
        lng_exports: Optional[pd.Series] = None,
        hdf_path: str = r"F:\Data\ng_eia_cache.hdf",
    ) -> None:
        """Save EIA data to HDF5 cache for fast reloads."""
        if storage is not None:
            NatGasStorageForecaster._hdf_save(storage, hdf_path, "ng/storage", "storage")
        if prices is not None:
            prices_df = prices.to_frame() if isinstance(prices, pd.Series) else prices
            NatGasStorageForecaster._hdf_save(prices_df, hdf_path, "ng/prices", "prices")
        if lng_exports is not None:
            lng_df = lng_exports.to_frame() if isinstance(lng_exports, pd.Series) else lng_exports
            NatGasStorageForecaster._hdf_save(lng_df, hdf_path, "ng/lng_exports", "LNG exports")
