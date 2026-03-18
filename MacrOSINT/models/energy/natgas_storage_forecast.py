"""
Natural Gas Storage Forecaster.

Uses population-weighted Heating/Cooling Degree Days from PopulationWeatherGrid
combined with EIA weekly underground storage data to forecast weekly storage
changes via SARIMAX regression.

References:
    - Gas-Weighted Degree Days (GWDD) methodology
    - SARIMAX(1,1,1)(1,1,1,52) with exogenous weather + price variables
    - Piecewise HDD for non-linear cold-weather demand response
"""
import warnings
from datetime import date, timedelta
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX

from MacrOSINT.data.sources.eia.api_tools import NatGasHelper
from MacrOSINT.models.weather.population_weather import (
    CENSUS_FILES,
    PopulationWeatherGrid,
    grid_epoch_year,
)

# Default base temperature for degree-day calculation (Fahrenheit)
BASE_TEMP_F = 65.0
# Celsius equivalent
BASE_TEMP_C = (BASE_TEMP_F - 32) * 5 / 9  # ~18.33

# Piecewise HDD threshold: degrees below base that separates mild from extreme
MILD_HDD_CUTOFF = 15.0  # Fahrenheit


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
    Compute HDD, CDD, and piecewise HDD from a population-weighted daily
    temperature series.

    Args:
        daily_temps: DataFrame with DatetimeIndex and a weighted-average
                     temperature column (from PopulationWeatherGrid).
        base_temp: Base temperature for degree-day calc.
        temp_col: Column name containing the weighted temperature.
        celsius: If True, temperatures and base are in Celsius.

    Returns:
        DataFrame with columns: HDD, CDD, HDD_mild, HDD_extreme
    """
    if temp_col not in daily_temps.columns:
        raise KeyError(f"Column '{temp_col}' not found. Available: {list(daily_temps.columns)}")

    tavg = daily_temps[temp_col].copy()

    # Convert cutoff to same unit as temperatures
    if celsius:
        cutoff_c = MILD_HDD_CUTOFF * 5 / 9  # ~8.33 C
    else:
        cutoff_c = MILD_HDD_CUTOFF

    hdd = (base_temp - tavg).clip(lower=0)
    cdd = (tavg - base_temp).clip(lower=0)

    hdd_mild = hdd.clip(upper=cutoff_c)
    hdd_extreme = (hdd - cutoff_c).clip(lower=0)

    return pd.DataFrame({
        "HDD": hdd,
        "CDD": cdd,
        "HDD_mild": hdd_mild,
        "HDD_extreme": hdd_extreme,
    }, index=daily_temps.index)


def resample_weekly(degree_days: pd.DataFrame, agg: str = "sum") -> pd.DataFrame:
    """Resample daily degree days to weekly (Friday-ending to match EIA)."""
    if agg == "sum":
        return degree_days.resample("W-FRI").sum()
    return degree_days.resample("W-FRI").mean()


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

    Returns:
        DataFrame indexed by date with columns:
        - storage_level: total working gas in storage (Bcf)
        - storage_change: weekly change in storage (Bcf)
    """
    raw = ng_helper.get_underground_storage(start=start, end=end)
    if raw is None or raw.empty:
        raise ValueError("No storage data returned from EIA API")

    # clean_api_data returns 'period' as a column with reset_index=True
    if "period" in raw.columns:
        raw["period"] = pd.to_datetime(raw["period"])
        raw = raw.set_index("period").sort_index()

    # Drop non-numeric columns
    raw = raw.drop(columns=["units"], errors="ignore")

    # Use R48 (Total lower 48) if available, else sum regions
    if "R48" in raw.columns:
        storage = raw["R48"].astype(float)
    else:
        numeric_cols = raw.select_dtypes(include="number").columns
        storage = raw[numeric_cols].sum(axis=1)

    df = pd.DataFrame({"storage_level": storage}, index=raw.index)
    df.index.name = "date"
    df["storage_change"] = df["storage_level"].diff()
    df = df.dropna(subset=["storage_change"])
    return df


def fetch_spot_prices(
    ng_helper: NatGasHelper,
    start: Optional[str] = None,
    end: Optional[str] = None,
) -> pd.Series:
    """
    Fetch Henry Hub natural gas spot/futures prices from EIA and resample
    to weekly (Friday) averages.
    """
    raw = ng_helper.execute_request("spot_prices", start=start, end=end)
    if raw is None or raw.empty:
        warnings.warn("No spot price data returned; model will run without price feature.")
        return pd.Series(dtype=float)

    # clean_api_data returns 'period' as a column with reset_index=True
    if "period" in raw.columns:
        raw["period"] = pd.to_datetime(raw["period"])
        raw = raw.set_index("period").sort_index()

    raw = raw.drop(columns=["units"], errors="ignore")

    # Use first numeric column (EPG0 = Henry Hub)
    numeric_cols = raw.select_dtypes(include="number").columns
    if len(numeric_cols) == 0:
        warnings.warn("No numeric price columns found.")
        return pd.Series(dtype=float)

    price_col = numeric_cols[0]
    weekly = raw[price_col].astype(float).resample("W-FRI").mean()
    weekly.name = "gas_price"
    return weekly


# ---------------------------------------------------------------------------
# Forecaster class
# ---------------------------------------------------------------------------

class NatGasStorageForecaster:
    """
    Weekly natural gas storage change forecaster using population-weighted
    degree days and SARIMAX regression.

    Usage:
        forecaster = NatGasStorageForecaster()
        forecaster.setup()                         # build weather grid + fetch data
        forecaster.fit(start='2020-01', end='2025-12')
        forecast = forecaster.forecast(steps=4, future_weather=weather_df)
    """

    def __init__(
        self,
        weather_grid: Optional[PopulationWeatherGrid] = None,
        ng_helper: Optional[NatGasHelper] = None,
        ncei_token: Optional[str] = None,
        config_dir: Optional[str] = None,
        base_temp_c: float = BASE_TEMP_C,
        order: Tuple[int, int, int] = (1, 1, 1),
        seasonal_order: Tuple[int, int, int, int] = (1, 1, 1, 52),
        use_price: bool = True,
        use_piecewise_hdd: bool = True,
    ):
        self.ng_helper = ng_helper or NatGasHelper()
        self.ncei_token = ncei_token
        self.config_dir = Path(config_dir) if config_dir else None
        self.base_temp_c = base_temp_c
        self.order = order
        self.seasonal_order = seasonal_order
        self.use_price = use_price
        self.use_piecewise_hdd = use_piecewise_hdd

        # Grid registry: census_year -> PopulationWeatherGrid
        self._grids: dict = {}
        if weather_grid is not None:
            self._grids[weather_grid.census_year] = weather_grid
        self.weather_grid = weather_grid  # default / active grid

        # Fitted model
        self._model = None
        self._results = None
        self._training_data: Optional[pd.DataFrame] = None
        self._exog_cols: list = []

    # -- setup ---------------------------------------------------------------

    def _config_path_for_epoch(self, epoch_year: int) -> Optional[Path]:
        """Locate config file for a given census epoch."""
        if self.config_dir is None:
            return None
        candidates = [
            self.config_dir / f"pop_weather_config_{epoch_year}.json",
            self.config_dir / "pop_weather_config.json",
        ]
        for p in candidates:
            if p.exists():
                return p
        return None

    def get_grid(self, epoch_year: int) -> PopulationWeatherGrid:
        """Get or load the PopulationWeatherGrid for a census epoch."""
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
        """
        Initialize the weather grid. If a saved config exists, load it
        to skip the clustering/station-finding step.
        """
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
        """Fetch weighted daily weather, switching grids at epoch boundaries."""
        start_epoch = grid_epoch_year(start_dt.year)
        end_epoch = grid_epoch_year(end_dt.year)

        if start_epoch == end_epoch:
            grid = self.get_grid(start_epoch)
            return grid.get_weighted_daily(start_dt, end_dt, level="national")

        # Span crosses epoch boundary -- stitch segments
        epochs = sorted(CENSUS_FILES.keys())
        # Filter to epochs relevant for this date range
        relevant = [e for e in epochs if e <= end_epoch]
        if not relevant:
            relevant = [epochs[0]]

        frames = []
        for i, epoch in enumerate(relevant):
            if epoch < start_epoch:
                continue
            seg_start = max(start_dt, date(epoch, 1, 1))
            # Segment ends at the start of next epoch - 1 day, or end_dt
            if i + 1 < len(relevant):
                seg_end = min(end_dt, date(relevant[i + 1] - 1, 12, 31))
            else:
                seg_end = end_dt
            grid = self.get_grid(epoch)
            chunk = grid.get_weighted_daily(seg_start, seg_end, level="national")
            if not chunk.empty:
                frames.append(chunk)

        if not frames:
            return pd.DataFrame()
        daily = pd.concat(frames).sort_index()
        return daily[~daily.index.duplicated(keep="first")]

    def build_features(
        self,
        start: str,
        end: str,
        daily_weather: Optional[pd.DataFrame] = None,
    ) -> pd.DataFrame:
        """
        Assemble the weekly feature matrix: storage change (target),
        population-weighted HDD/CDD, and optionally spot price.

        Args:
            start: Start date string (e.g. '2020-01')
            end: End date string (e.g. '2025-12')
            daily_weather: Pre-fetched weighted daily temps (skips weather API).
                           Must have DatetimeIndex and a 'wtd_TAVG' column.

        Returns:
            DataFrame indexed by weekly date with target + exogenous columns.
        """
        start_dt = pd.Timestamp(start).date()
        end_dt = pd.Timestamp(end).date()

        # 1. Population-weighted daily temperatures (national level)
        if daily_weather is not None:
            daily = daily_weather
        else:
            daily = self._fetch_weather_by_epoch(start_dt, end_dt)
        if daily.empty:
            raise ValueError("No weather data returned for the requested period")

        # 2. Degree days
        dd = compute_degree_days(daily, base_temp=self.base_temp_c)
        weekly_dd = resample_weekly(dd)

        # 3. Storage data
        storage = fetch_storage_data(self.ng_helper, start=start, end=end)

        # 4. Spot prices (optional)
        if self.use_price:
            prices = fetch_spot_prices(self.ng_helper, start=start, end=end)
        else:
            prices = pd.Series(dtype=float)

        # 5. Merge on weekly index
        merged = storage.join(weekly_dd, how="inner")
        if not prices.empty:
            merged = merged.join(prices, how="left")
            merged["gas_price"] = merged["gas_price"].ffill()

        return merged.dropna()

    def _select_exog_cols(self, df: pd.DataFrame) -> list:
        """Pick exogenous columns based on configuration."""
        cols = []
        if self.use_piecewise_hdd:
            cols.extend(["HDD_mild", "HDD_extreme"])
        else:
            cols.append("HDD")
        cols.append("CDD")
        if self.use_price and "gas_price" in df.columns:
            cols.append("gas_price")
        return [c for c in cols if c in df.columns]

    # -- model ---------------------------------------------------------------

    def fit(
        self,
        start: str = "2020-01",
        end: str = None,
        data: Optional[pd.DataFrame] = None,
    ) -> "NatGasStorageForecaster":
        """
        Fit the SARIMAX model on historical data.

        Args:
            start: Training period start.
            end: Training period end (default: today).
            data: Pre-built feature DataFrame. If None, calls build_features().
        """
        if end is None:
            end = date.today().strftime("%Y-%m")

        if data is None:
            data = self.build_features(start, end)

        self._training_data = data
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

    def summary(self):
        """Return statsmodels summary of the fitted model."""
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
            steps: Number of weeks ahead.
            future_exog: DataFrame with exogenous values for forecast horizon.
                         Must have the same columns as training exog.
                         If None and model has exog, raises an error.

        Returns:
            DataFrame with columns: forecast, lower_ci, upper_ci
        """
        if self._results is None:
            raise RuntimeError("Model not fitted. Call fit() first.")

        if self._exog_cols and future_exog is None:
            raise ValueError(
                f"Model was trained with exogenous variables {self._exog_cols}. "
                "Provide future_exog DataFrame for forecasting."
            )

        exog = future_exog[self._exog_cols] if future_exog is not None else None

        fc = self._results.get_forecast(steps=steps, exog=exog)
        pred = fc.predicted_mean
        ci = fc.conf_int(alpha=0.05)

        result = pd.DataFrame({
            "forecast": pred.values,
            "lower_ci": ci.iloc[:, 0].values,
            "upper_ci": ci.iloc[:, 1].values,
        }, index=pred.index)

        return result

    def forecast_storage_levels(
        self,
        steps: int = 4,
        future_exog: Optional[pd.DataFrame] = None,
    ) -> pd.DataFrame:
        """
        Forecast absolute storage levels by accumulating predicted changes
        on top of the last observed level.
        """
        changes = self.forecast(steps=steps, future_exog=future_exog)
        last_level = self._training_data["storage_level"].iloc[-1]

        changes["level_forecast"] = last_level + changes["forecast"].cumsum()
        changes["level_lower"] = last_level + changes["lower_ci"].cumsum()
        changes["level_upper"] = last_level + changes["upper_ci"].cumsum()

        return changes

    # -- diagnostics ---------------------------------------------------------

    def insample_fit(self) -> pd.DataFrame:
        """Return in-sample fitted values alongside actuals for diagnostics."""
        if self._results is None:
            raise RuntimeError("Model not fitted. Call fit() first.")

        fitted = self._results.fittedvalues
        actual = self._training_data["storage_change"]

        return pd.DataFrame({
            "actual": actual,
            "fitted": fitted,
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
        """Save population-weighted daily weather to HDF5 store."""
        path = Path(hdf_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        daily_weather.to_hdf(str(path), key=key, mode="a", format="table")
        print(f"Saved weather data to {path} [key={key}] ({len(daily_weather)} rows)")

    @staticmethod
    def load_weather_hdf(
        hdf_path: str = r"F:\Data\weather.hdf",
        key: str = "weather/ng/storage_weather_data",
    ) -> Optional[pd.DataFrame]:
        """Load population-weighted daily weather from HDF5 store."""
        path = Path(hdf_path)
        if not path.exists():
            return None
        try:
            df = pd.read_hdf(str(path), key=key)
            df.index = pd.to_datetime(df.index)
            return df
        except KeyError:
            return None
