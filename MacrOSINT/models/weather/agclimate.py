"""
Agricultural Climate Data Analysis Module (agclimate)
======================================================

A comprehensive Python module for accessing and analyzing climate data from
NOAA's GHCND (daily) and GSOM (monthly) datasets for agricultural applications.

Author: Agricultural Climate Analysis Team
Version: 1.0.0
License: MIT

Requirements:
    - pyncei
    - pandas
    - numpy
    - geopandas (optional, for spatial analysis)
    - scipy

Usage:
    from agclimate import AgClimateAPI, CropClimateAnalyzer

    # Initialize API
    api = AgClimateAPI(token='your_ncei_token')

    # Get daily data for corn belt
    daily_data = api.get_agricultural_data(
        dataset='GHCND',
        region='corn_belt',
        start_date=date(2020, 1, 1),
        end_date=date(2023, 12, 31)
    )

    # Analyze for specific crop
    analyzer = CropClimateAnalyzer()
    corn_analysis = analyzer.analyze_crop_climate(daily_data, crop='corn')
"""

import os
import time
import warnings
from datetime import date, datetime
from typing import List, Dict, Tuple, Optional, Union
from dataclasses import dataclass
from enum import Enum

import pandas as pd
import numpy as np
from pyncei import NCEIBot

# Optional imports
try:
    import geopandas as gpd

    HAS_GEOPANDAS = True
except ImportError:
    HAS_GEOPANDAS = False
    warnings.warn("geopandas not installed. Spatial features will be limited.")


# =============================================================================
# Configuration and Constants
# =============================================================================

class DatasetType(Enum):
    """Supported NCEI dataset types."""
    GHCND = "GHCND"  # Global Historical Climatology Network - Daily
    GSOM = "GSOM"  # Global Summary of the Month
    GSOY = "GSOY"  # Global Summary of the Year
    NORMAL_DLY = "NORMAL_DLY"  # Daily climate normals
    NORMAL_MLY = "NORMAL_MLY"  # Monthly climate normals


@dataclass
class AgriculturalRegion:
    """Define agricultural regions with metadata."""
    name: str
    extent: str  # "min_lat,min_lon,max_lat,max_lon"
    states: List[str]
    primary_crops: List[str]
    description: str
    climate_divisions: Optional[List[int]] = None


@dataclass
class CropParameters:
    """Crop-specific climate parameters."""
    name: str
    base_temp: float  # Base temperature for GDD calculation (°C)
    max_temp: float  # Maximum temperature for GDD calculation (°C)
    start_month: int
    end_month: int
    optimal_temp_range: Tuple[float, float]  # (min, max) in °C
    optimal_precip_range: Tuple[float, float]  # (min, max) in mm for growing season
    critical_months: List[int]
    frost_sensitive: bool = True
    drought_tolerance: str = "moderate"  # low, moderate, high


# =============================================================================
# Regional and Crop Definitions
# =============================================================================

AGRICULTURAL_REGIONS = {
    'corn_belt': AgriculturalRegion(
        name='Corn Belt',
        extent='38.0,-100.0,45.0,-85.0',
        states=['IA', 'IL', 'IN', 'OH', 'MO', 'NE', 'MN', 'SD', 'WI'],
        primary_crops=['corn', 'soybeans'],
        description='Midwest corn and soybean production region',
        climate_divisions=[1301, 1302, 1303, 1304, 1305, 1306, 1307, 1308, 1309]  # Iowa divisions
    ),
    'wheat_belt': AgriculturalRegion(
        name='Wheat Belt',
        extent='35.0,-110.0,50.0,-95.0',
        states=['KS', 'ND', 'MT', 'OK', 'NE', 'CO', 'SD', 'TX'],
        primary_crops=['wheat', 'sorghum'],
        description='Great Plains wheat production region'
    ),
    'cotton_belt': AgriculturalRegion(
        name='Cotton Belt',
        extent='30.0,-105.0,37.0,-75.0',
        states=['TX', 'GA', 'MS', 'AL', 'AR', 'LA', 'NC', 'SC', 'TN'],
        primary_crops=['cotton', 'peanuts'],
        description='Southern cotton production region'
    ),
    'california_central_valley': AgriculturalRegion(
        name='California Central Valley',
        extent='35.0,-122.5,40.5,-118.5',
        states=['CA'],
        primary_crops=['almonds', 'grapes', 'tomatoes', 'rice'],
        description='California intensive agriculture region'
    ),
    'pacific_northwest': AgriculturalRegion(
        name='Pacific Northwest',
        extent='42.0,-125.0,49.0,-116.0',
        states=['WA', 'OR', 'ID'],
        primary_crops=['wheat', 'potatoes', 'apples'],
        description='Pacific Northwest agricultural region'
    )
}

CROP_PARAMETERS = {
    'corn': CropParameters(
        name='Corn/Maize',
        base_temp=10.0,
        max_temp=30.0,
        start_month=4,
        end_month=10,
        optimal_temp_range=(18.0, 32.0),
        optimal_precip_range=(500.0, 800.0),
        critical_months=[7, 8],  # Pollination and grain fill
        frost_sensitive=True,
        drought_tolerance='moderate'
    ),
    'soybeans': CropParameters(
        name='Soybeans',
        base_temp=10.0,
        max_temp=30.0,
        start_month=5,
        end_month=10,
        optimal_temp_range=(20.0, 30.0),
        optimal_precip_range=(450.0, 700.0),
        critical_months=[7, 8, 9],  # Pod development and fill
        frost_sensitive=True,
        drought_tolerance='moderate'
    ),
    'wheat_winter': CropParameters(
        name='Winter Wheat',
        base_temp=0.0,
        max_temp=26.0,
        start_month=10,
        end_month=6,
        optimal_temp_range=(12.0, 25.0),
        optimal_precip_range=(400.0, 600.0),
        critical_months=[4, 5, 6],  # Heading and grain fill
        frost_sensitive=False,
        drought_tolerance='high'
    ),
    'wheat_spring': CropParameters(
        name='Spring Wheat',
        base_temp=0.0,
        max_temp=26.0,
        start_month=3,
        end_month=8,
        optimal_temp_range=(15.0, 25.0),
        optimal_precip_range=(350.0, 550.0),
        critical_months=[6, 7],  # Heading and grain fill
        frost_sensitive=True,
        drought_tolerance='moderate'
    ),
    'cotton': CropParameters(
        name='Cotton',
        base_temp=15.6,
        max_temp=35.0,
        start_month=4,
        end_month=11,
        optimal_temp_range=(20.0, 30.0),
        optimal_precip_range=(500.0, 1200.0),
        critical_months=[7, 8, 9],  # Boll development
        frost_sensitive=True,
        drought_tolerance='high'
    ),
    'rice': CropParameters(
        name='Rice',
        base_temp=10.0,
        max_temp=35.0,
        start_month=4,
        end_month=9,
        optimal_temp_range=(20.0, 35.0),
        optimal_precip_range=(1200.0, 2000.0),
        critical_months=[6, 7, 8],  # Flowering and grain fill
        frost_sensitive=True,
        drought_tolerance='low'
    )
}


# =============================================================================
# Main API Class
# =============================================================================

class AgClimateAPI:
    """
    Main API class for accessing NCEI climate data for agricultural applications.

    Supports both GHCND (daily) and GSOM (monthly) datasets with unified interface.
    """

    def __init__(self, token: Optional[str] = None, cache_name: str = "agclimate_cache",
                 cache_duration_hours: int = 24):
        """
        Initialize the Agricultural Climate API.

        Args:
            token: NCEI API token (get from https://www.ncdc.noaa.gov/cdo-web/token)
            cache_name: Name for cache database
            cache_duration_hours: Cache expiration time in hours
        """
        from MacrOSINT.config import NCEI_TOKEN
        self.token = token or NCEI_TOKEN
        if not self.token:
            warnings.warn("No API token provided. Some features may be limited.")

        self.ncei = NCEIBot(
            self.token,
            cache_name=cache_name,
            expire_after=cache_duration_hours * 3600
        ) if self.token else None

        self.regions = AGRICULTURAL_REGIONS
        self.crop_params = CROP_PARAMETERS

        # Data quality flags to exclude
        self.problem_flags = ['D', 'G', 'I', 'K', 'M', 'N', 'O', 'R', 'S', 'T', 'X', 'Z']

        # Variable mappings for different datasets
        self.variable_mappings = {
            'GHCND': {
                'temperature_min': 'TMIN',
                'temperature_max': 'TMAX',
                'temperature_avg': 'TAVG',
                'precipitation': 'PRCP',
                'snowfall': 'SNOW',
                'snow_depth': 'SNWD'
            },
            'GSOM': {
                'temperature_min': 'TMIN',
                'temperature_max': 'TMAX',
                'temperature_avg': 'TAVG',
                'precipitation': 'TPCP',
                'heating_degree_days': 'HTDD',
                'cooling_degree_days': 'CLDD'
            }
        }

    def get_agricultural_data(self, dataset: str = 'GHCND', region: Union[str, AgriculturalRegion] = None,
                              start_date: date = None, end_date: date = None,
                              variables: List[str] = None, extent: str = None,
                              min_coverage: float = 0.8, max_stations: int = 20) -> pd.DataFrame:
        """
        Get climate data for agricultural analysis from either GHCND or GSOM.

        Args:
            dataset: Dataset type ('GHCND' for daily, 'GSOM' for monthly)
            region: Agricultural region name or AgriculturalRegion object
            start_date: Start date for data retrieval
            end_date: End date for data retrieval
            variables: Climate variables to retrieve (uses defaults if None)
            extent: Custom extent as "min_lat,min_lon,max_lat,max_lon"
            min_coverage: Minimum data coverage threshold (0.0-1.0)
            max_stations: Maximum number of stations to retrieve

        Returns:
            DataFrame with climate data
        """
        if not self.ncei:
            raise ValueError("API token required for data retrieval")

        # Handle region specification
        region_obj = None
        if isinstance(region, str):
            if region not in self.regions:
                raise ValueError(f"Unknown region: {region}. Available: {list(self.regions.keys())}")
            region_obj = self.regions[region]
            extent = region_obj.extent
        elif isinstance(region, AgriculturalRegion):
            region_obj = region
            extent = region_obj.extent
        elif not extent:
            raise ValueError("Either region or extent must be specified")

        # Set default variables based on dataset
        if variables is None:
            if dataset == 'GHCND':
                variables = ['TMIN', 'TMAX', 'PRCP', 'TAVG']
            elif dataset == 'GSOM':
                variables = ['TMIN', 'TMAX', 'TAVG', 'TPCP']
            else:
                raise ValueError(f"Unsupported dataset: {dataset}")

        # Get stations
        print(f"Finding {dataset} stations in region...")
        stations = self._get_quality_stations(
            dataset, extent, start_date, end_date,
            variables, min_coverage, max_stations
        )

        if stations.empty:
            warnings.warn(f"No stations found for specified criteria")
            return pd.DataFrame()

        print(f"Found {len(stations)} stations. Retrieving data...")

        # Retrieve data based on dataset type
        if dataset == 'GHCND':
            data = self._get_ghcnd_data(stations, start_date, end_date, variables)
        elif dataset == 'GSOM':
            data = self._get_gsom_data(stations, start_date, end_date, variables)
        else:
            raise ValueError(f"Unsupported dataset: {dataset}")

        # Add metadata
        if not data.empty:
            data['dataset'] = dataset
            if region_obj and hasattr(region_obj, 'name'):
                data['region'] = region_obj.name

        return data

    def _get_quality_stations(self, dataset: str, extent: str, start_date: date,
                              end_date: date, variables: List[str],
                              min_coverage: float, max_stations: int) -> pd.DataFrame:
        """Get stations with quality filtering.

        Stations are selected in a round-robin manner from different parts of
        the requested region. The order of selection is:

        north → south → center → east → west and repeats until the
        ``max_stations`` limit has been reached or no stations remain.
        """
        try:
            stations_response = self.ncei.get_stations(
                datasetid=dataset,
                datatypeid=variables,
                extent=extent,
                startdate=start_date,
                enddate=end_date,
                limit=1000
            )

            stations_df = stations_response.to_dataframe()

            # Filter by data coverage
            quality_stations = stations_df[
                stations_df['datacoverage'] >= min_coverage
            ].copy()

            if quality_stations.empty:
                return quality_stations

            # Parse extent and compute distances to region boundaries/center
            min_lat, min_lon, max_lat, max_lon = map(float, extent.split(','))
            center_lat = (min_lat + max_lat) / 2
            center_lon = (min_lon + max_lon) / 2

            quality_stations['north_dist'] = max_lat - quality_stations['latitude']
            quality_stations['south_dist'] = quality_stations['latitude'] - min_lat
            quality_stations['east_dist'] = max_lon - quality_stations['longitude']
            quality_stations['west_dist'] = quality_stations['longitude'] - min_lon
            quality_stations['center_dist'] = np.sqrt(
                (quality_stations['latitude'] - center_lat) ** 2 +
                (quality_stations['longitude'] - center_lon) ** 2
            )

            # Prepare sorted lists for each direction
            direction_lists = {
                'north': quality_stations.sort_values(
                    ['north_dist', 'datacoverage'], ascending=[True, False]
                ),
                'south': quality_stations.sort_values(
                    ['south_dist', 'datacoverage'], ascending=[True, False]
                ),
                'central': quality_stations.sort_values(
                    ['center_dist', 'datacoverage'], ascending=[True, False]
                ),
                'east': quality_stations.sort_values(
                    ['east_dist', 'datacoverage'], ascending=[True, False]
                ),
                'west': quality_stations.sort_values(
                    ['west_dist', 'datacoverage'], ascending=[True, False]
                ),
            }

            indices = {key: 0 for key in direction_lists.keys()}
            selected_rows = []
            selected_ids = set()
            order = ['central', 'south', 'north', 'east', 'west']

            while len(selected_rows) < max_stations:
                added_any = False
                for direction in order:
                    stations_list = direction_lists[direction]
                    idx = indices[direction]

                    # Skip already-selected stations
                    while idx < len(stations_list) and stations_list.iloc[idx]['id'] in selected_ids:
                        idx += 1

                    indices[direction] = idx

                    if idx < len(stations_list) and len(selected_rows) < max_stations:
                        selected_rows.append(stations_list.iloc[idx])
                        selected_ids.add(stations_list.iloc[idx]['id'])
                        indices[direction] += 1
                        added_any = True

                    if len(selected_rows) >= max_stations:
                        break

                if not added_any:
                    break

            selected_df = pd.DataFrame(selected_rows)
            drop_cols = ['north_dist', 'south_dist', 'east_dist', 'west_dist', 'center_dist']
            return selected_df.drop(columns=drop_cols, errors='ignore').reset_index(drop=True)

        except Exception as e:
            print(f"Error getting stations: {e}")
            return pd.DataFrame()

    def _get_ghcnd_data(self, stations: pd.DataFrame, start_date: date,
                        end_date: date, variables: List[str]) -> pd.DataFrame:
        """Retrieve GHCND daily data with year-by-year chunking."""
        all_data = []
        successful_stations = 0
        total_stations = len(stations)

        for idx, station in stations.iterrows():
            station_id = station['id']
            print(f"  Processing {station['name']} ({station_id})...")
            
            station_data = []
            successful_years = 0
            total_years = end_date.year - start_date.year + 1

            # Process year by year due to API limitations
            current_year = start_date.year
            while current_year <= end_date.year:
                year_start = max(start_date, date(current_year, 1, 1))
                year_end = min(end_date, date(current_year, 12, 31))

                try:
                    year_data = self.ncei.get_data(
                        datasetid='GHCND',
                        stationid=station_id,
                        datatypeid=variables,
                        startdate=year_start,
                        enddate=year_end,
                        units='metric'
                    )

                    if year_data:
                        df = year_data.to_dataframe()
                        if not df.empty:
                            df['station_name'] = station['name']
                            df['latitude'] = station.get('latitude', None)
                            df['longitude'] = station.get('longitude', None)
                            df['elevation'] = station.get('elevation', None)
                            station_data.append(df)
                            successful_years += 1

                    time.sleep(0.2)  # Rate limiting

                except Exception as e:
                    print(f"    Error retrieving {current_year} data: {e}")
                    # Continue to next year instead of failing entire station

                current_year += 1
            
            # Add station data if we got at least some years successfully
            if station_data:
                all_data.extend(station_data)
                successful_stations += 1
                print(f"    SUCCESS: Retrieved {successful_years}/{total_years} years of data")
            else:
                print(f"    WARNING: No data retrieved for station {station_id}")

        print(f"  Station Summary: {successful_stations}/{total_stations} stations returned data")
        
        if all_data:
            combined_data = pd.concat(all_data, ignore_index=True)
            return self._process_raw_data(combined_data, 'GHCND')

        return pd.DataFrame()

    def _get_gsom_data(self, stations: pd.DataFrame, start_date: date,
                       end_date: date, variables: List[str]) -> pd.DataFrame:
        """Retrieve GSOM monthly data."""
        all_data = []
        successful_stations = 0
        total_stations = len(stations)

        for idx, station in stations.iterrows():
            station_id = station['id']
            print(f"  Processing {station['name']} ({station_id})...")

            try:
                # GSOM can handle longer date ranges
                monthly_data = self.ncei.get_data(
                    datasetid='GSOM',
                    stationid=station_id,
                    datatypeid=variables,
                    startdate=start_date,
                    enddate=end_date,
                    units='metric'
                )

                if monthly_data:
                    df = monthly_data.to_dataframe()
                    if not df.empty:
                        df['station_name'] = station['name']
                        df['latitude'] = station.get('latitude', None)
                        df['longitude'] = station.get('longitude', None)
                        df['elevation'] = station.get('elevation', None)
                        all_data.append(df)
                        successful_stations += 1
                        print(f"    SUCCESS: Retrieved {len(df)} records")
                    else:
                        print(f"    WARNING: Empty dataframe returned for station {station_id}")
                else:
                    print(f"    WARNING: No data returned for station {station_id}")

                time.sleep(0.2)  # Rate limiting

            except Exception as e:
                print(f"    ERROR: Failed to retrieve data for station {station_id}: {e}")
                # Continue to next station instead of failing entire operation
                continue

        print(f"  Station Summary: {successful_stations}/{total_stations} stations returned data")
        
        if all_data:
            combined_data = pd.concat(all_data, ignore_index=True)
            return self._process_raw_data(combined_data, 'GSOM')

        return pd.DataFrame()

    def _process_raw_data(self, df: pd.DataFrame, dataset: str) -> pd.DataFrame:
        """Process raw data with quality control and unit conversion."""
        df = df.copy()

        # Convert date column
        df['date'] = pd.to_datetime(df['date'])

        # Handle missing values
        df['value'] = df['value'].replace(-9999, np.nan)
        df['value'] = df['value'].replace(-999.9, np.nan)

        # Extract quality flags if available
        if 'attributes' in df.columns:
            df['qflag'] = df['attributes'].apply(
                lambda x: x.split(',')[0] if isinstance(x, str) and ',' in x else ''
            )
            df['quality_issue'] = df['qflag'].isin(self.problem_flags)
        else:
            df['quality_issue'] = False

        # Unit conversions based on dataset and variable
        if dataset == 'GHCND':
            # GHCND temperature is in tenths of degrees Celsius
            temp_vars = ['TMIN', 'TMAX', 'TAVG']
            for var in temp_vars:
                mask = df['datatype'] == var
                df.loc[mask, 'value'] = df.loc[mask, 'value']

            # GHCND precipitation is in tenths of mm
            precip_mask = df['datatype'] == 'PRCP'
            df.loc[precip_mask, 'value'] = df.loc[precip_mask, 'value']

            # Snow depth is in mm (no conversion needed)

        elif dataset == 'GSOM':
            # GSOM data typically comes in standard units after API conversion
            # But verify and adjust if needed
            pass

        # Add time components
        df['year'] = df['date'].dt.year
        df['month'] = df['date'].dt.month
        if dataset == 'GHCND':
            df['day'] = df['date'].dt.day
            df['doy'] = df['date'].dt.dayofyear

        # Sort by station, date, and datatype
        df = df.sort_values(['station', 'date', 'datatype'])

        return df

    def get_station_metadata(self, dataset: str = 'GHCND',
                             region: str = None) -> pd.DataFrame:
        """
        Get metadata for stations in a region.

        Args:
            dataset: Dataset type
            region: Agricultural region name

        Returns:
            DataFrame with station metadata
        """
        if not self.ncei:
            raise ValueError("API token required")

        if region and region in self.regions:
            extent = self.regions[region].extent
        else:
            extent = None

        stations = self.ncei.get_stations(
            datasetid=dataset,
            extent=extent,
            limit=1000
        )

        return stations.to_dataframe() if stations else pd.DataFrame()


# =============================================================================
# Data Quality and Processing
# =============================================================================

class DataQualityProcessor:
    """Handle data quality control and missing value imputation."""

    def __init__(self):
        self.temp_bounds = {'min': -50, 'max': 60}  # Celsius
        self.precip_max = 500  # mm/day maximum reasonable precipitation
        self.problem_flags = ['D', 'G', 'I', 'K', 'M', 'N', 'O', 'R', 'S', 'T', 'X', 'Z']

    def assess_quality(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Comprehensive quality assessment for agricultural data.

        Args:
            df: Input dataframe with climate data

        Returns:
            DataFrame with quality assessments added
        """
        df = df.copy()

        # Initialize quality score (0-100)
        df['quality_score'] = 100

        # Check for quality flags
        if 'quality_issue' in df.columns:
            df.loc[df['quality_issue'] == True, 'quality_score'] -= 20

        # Check for extreme values
        temp_vars = ['TMIN', 'TMAX', 'TAVG']
        for var in temp_vars:
            if var in df['datatype'].values:
                mask = df['datatype'] == var
                extreme_mask = mask & ((df['value'] < self.temp_bounds['min']) |
                                       (df['value'] > self.temp_bounds['max']))
                df.loc[extreme_mask, 'quality_score'] -= 30
                df.loc[extreme_mask, 'quality_note'] = 'extreme_temperature'

        # Check precipitation extremes
        if 'PRCP' in df['datatype'].values:
            precip_mask = df['datatype'] == 'PRCP'
            extreme_precip = precip_mask & (df['value'] > self.precip_max)
            df.loc[extreme_precip, 'quality_score'] -= 30
            df.loc[extreme_precip, 'quality_note'] = 'extreme_precipitation'

        # Check for logical consistency (TMIN <= TMAX)
        if all(var in df['datatype'].values for var in ['TMIN', 'TMAX']):
            pivot_temps = df[df['datatype'].isin(['TMIN', 'TMAX'])].pivot_table(
                index=['station', 'date'], columns='datatype', values='value'
            )
            if not pivot_temps.empty:
                inconsistent = pivot_temps['TMIN'] > pivot_temps['TMAX']
                inconsistent_dates = inconsistent[inconsistent].index
                for idx in inconsistent_dates:
                    mask = (df['station'] == idx[0]) & (df['date'] == idx[1])
                    df.loc[mask, 'quality_score'] -= 40
                    df.loc[mask, 'quality_note'] = 'tmin_exceeds_tmax'

        return df

    def fill_missing_values(self, df: pd.DataFrame, method: str = 'interpolation',
                            max_gap: int = 3) -> pd.DataFrame:
        """
        Fill missing values using appropriate methods for agricultural data.

        Args:
            df: Input dataframe
            method: Method for filling ('interpolation', 'climatology', 'forward_fill')
            max_gap: Maximum gap size to fill (days for GHCND, months for GSOM)

        Returns:
            DataFrame with filled values
        """
        df_filled = df.copy()
        df_filled['filled'] = False

        # Group by station and datatype for filling
        for (station, datatype), group in df_filled.groupby(['station', 'datatype']):
            group = group.sort_values('date')
            original_values = group['value'].copy()

            if method == 'interpolation':
                # Linear interpolation for small gaps
                filled_values = group['value'].interpolate(
                    method='linear',
                    limit=max_gap,
                    limit_direction='both'
                )

            elif method == 'climatology':
                # Use historical averages
                if 'doy' in group.columns:  # Daily data
                    climatology = group.groupby('doy')['value'].transform('mean')
                else:  # Monthly data
                    climatology = group.groupby('month')['value'].transform('mean')

                filled_values = group['value'].fillna(climatology)

            elif method == 'forward_fill':
                # Forward fill for conservative approach
                filled_values = group['value'].fillna(method='ffill', limit=max_gap)

            else:
                filled_values = group['value']

            # Mark filled values
            filled_mask = original_values.isna() & filled_values.notna()

            # Update main dataframe
            df_filled.loc[group.index, 'value'] = filled_values
            df_filled.loc[group.index[filled_mask], 'filled'] = True

        return df_filled


# =============================================================================
# Agricultural Analysis
# =============================================================================

class CropClimateAnalyzer:
    """Analyze climate data for specific crops and agricultural applications."""

    def __init__(self):
        self.crop_params = CROP_PARAMETERS
        self.quality_processor = DataQualityProcessor()

    def analyze_crop_climate(self, df: pd.DataFrame, crop: str,
                             include_stress_indices: bool = True) -> Dict:
        """
        Comprehensive climate analysis for a specific crop.

        Args:
            df: Climate data DataFrame
            crop: Crop type (must be in CROP_PARAMETERS)
            include_stress_indices: Whether to calculate stress indices

        Returns:
            Dictionary with analysis results
        """
        if crop not in self.crop_params:
            raise ValueError(f"Unknown crop: {crop}. Available: {list(self.crop_params.keys())}")

        params = self.crop_params[crop]
        results = {
            'crop': crop,
            'parameters': params,
            'data_summary': self._summarize_data(df),
            'growing_season_stats': None,
            'gdd_analysis': None,
            'stress_indices': None,
            'suitability_assessment': None
        }

        # Check if we have daily or monthly data
        is_daily = 'day' in df.columns or 'doy' in df.columns

        # Filter to growing season
        growing_data = self._filter_growing_season(df, params)

        if not growing_data.empty:
            # Calculate growing season statistics
            results['growing_season_stats'] = self._calculate_season_stats(growing_data)

            # Calculate GDD if we have daily temperature data
            if is_daily and all(var in df['datatype'].values for var in ['TMIN', 'TMAX']):
                results['gdd_analysis'] = self.calculate_gdd(df, params)

            # Calculate stress indices if requested
            if include_stress_indices:
                results['stress_indices'] = self._calculate_stress_indices(growing_data, params)

            # Assess suitability
            results['suitability_assessment'] = self._assess_suitability(growing_data, params)

        return results

    def calculate_gdd(self, df: pd.DataFrame, params: CropParameters) -> pd.DataFrame:
        """
        Calculate Growing Degree Days (GDD) for a crop.

        Args:
            df: Daily climate data
            params: Crop parameters

        Returns:
            DataFrame with GDD calculations
        """
        # Pivot to get daily temperatures
        daily_temps = df[df['datatype'].isin(['TMAX', 'TMIN'])].pivot_table(
            index=['station', 'date'], columns='datatype', values='value'
        ).reset_index()

        if 'TMAX' not in daily_temps.columns or 'TMIN' not in daily_temps.columns:
            warnings.warn("Both TMAX and TMIN required for GDD calculation")
            return pd.DataFrame()

        # Apply temperature caps
        daily_temps['TMAX_adj'] = daily_temps['TMAX'].clip(upper=params.max_temp)
        daily_temps['TMIN_adj'] = daily_temps['TMIN'].clip(lower=params.base_temp)

        # Calculate daily GDD
        daily_temps['daily_mean'] = (daily_temps['TMAX_adj'] + daily_temps['TMIN_adj']) / 2
        daily_temps['gdd'] = np.maximum(daily_temps['daily_mean'] - params.base_temp, 0)

        # Add time components
        daily_temps['year'] = daily_temps['date'].dt.year
        daily_temps['month'] = daily_temps['date'].dt.month
        daily_temps['doy'] = daily_temps['date'].dt.dayofyear

        # Calculate cumulative GDD
        daily_temps['gdd_cumulative'] = daily_temps.groupby(['station', 'year'])['gdd'].cumsum()

        return daily_temps

    def _filter_growing_season(self, df: pd.DataFrame, params: CropParameters) -> pd.DataFrame:
        """Filter data to growing season months."""
        df = df.copy()

        # Handle cross-year growing seasons (e.g., winter wheat)
        if params.end_month < params.start_month:
            mask = (df['month'] >= params.start_month) | (df['month'] <= params.end_month)
        else:
            mask = (df['month'] >= params.start_month) & (df['month'] <= params.end_month)

        return df[mask]

    def _summarize_data(self, df: pd.DataFrame) -> Dict:
        """Create summary statistics of the data."""
        summary = {
            'total_records': len(df),
            'stations': df['station'].nunique() if 'station' in df.columns else 0,
            'date_range': (df['date'].min(), df['date'].max()) if 'date' in df.columns else None,
            'variables': list(df['datatype'].unique()) if 'datatype' in df.columns else [],
            'missing_percentage': (df['value'].isna().sum() / len(df)) * 100 if 'value' in df.columns else 0
        }

        # Add quality summary if available
        if 'quality_score' in df.columns:
            summary['avg_quality_score'] = df['quality_score'].mean()
            summary['low_quality_percentage'] = (df['quality_score'] < 50).sum() / len(df) * 100

        return summary

    def _calculate_season_stats(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate seasonal statistics."""
        stats = []

        # Group by station and year
        for (station, year), group in df.groupby(['station', 'year']):
            station_stats = {
                'station': station,
                'year': year,
                'data_count': len(group)
            }

            # Temperature statistics
            for temp_var in ['TMIN', 'TMAX', 'TAVG']:
                temp_data = group[group['datatype'] == temp_var]['value']
                if not temp_data.empty:
                    station_stats[f'{temp_var}_mean'] = temp_data.mean()
                    station_stats[f'{temp_var}_std'] = temp_data.std()
                    station_stats[f'{temp_var}_min'] = temp_data.min()
                    station_stats[f'{temp_var}_max'] = temp_data.max()

            # Precipitation statistics
            precip_data = group[group['datatype'].isin(['PRCP', 'TPCP'])]['value']
            if not precip_data.empty:
                station_stats['precip_total'] = precip_data.sum()
                station_stats['precip_days'] = (precip_data > 1).sum()  # Days with >1mm
                station_stats['precip_max_daily'] = precip_data.max()

            stats.append(station_stats)

        return pd.DataFrame(stats)

    def _calculate_stress_indices(self, df: pd.DataFrame, params: CropParameters) -> Dict:
        """Calculate agricultural stress indices."""
        indices = {}

        # Heat stress
        if 'TMAX' in df['datatype'].values:
            tmax_data = df[df['datatype'] == 'TMAX']['value']
            heat_threshold = params.optimal_temp_range[1]
            indices['heat_stress_days'] = (tmax_data > heat_threshold).sum()
            indices['extreme_heat_days'] = (tmax_data > heat_threshold + 5).sum()

        # Cold stress
        if 'TMIN' in df['datatype'].values:
            tmin_data = df[df['datatype'] == 'TMIN']['value']
            cold_threshold = params.optimal_temp_range[0]
            indices['cold_stress_days'] = (tmin_data < cold_threshold).sum()

            if params.frost_sensitive:
                indices['frost_days'] = (tmin_data <= 0).sum()
                indices['severe_frost_days'] = (tmin_data <= -2).sum()

        # Moisture stress
        if 'PRCP' in df['datatype'].values or 'TPCP' in df['datatype'].values:
            precip_var = 'PRCP' if 'PRCP' in df['datatype'].values else 'TPCP'
            precip_data = df[df['datatype'] == precip_var]['value']

            # Calculate consecutive dry days
            is_dry = precip_data < 1  # Less than 1mm
            dry_streaks = is_dry.groupby((is_dry != is_dry.shift()).cumsum()).sum()
            indices['max_dry_streak'] = dry_streaks.max() if not dry_streaks.empty else 0
            indices['dry_days_total'] = is_dry.sum()

        return indices

    def _assess_suitability(self, df: pd.DataFrame, params: CropParameters) -> Dict:
        """Assess climate suitability for crop production."""
        suitability = {
            'overall_score': 100,
            'temperature_suitable': False,
            'precipitation_suitable': False,
            'stress_level': 'low',
            'limiting_factors': []
        }

        # Temperature assessment
        if 'TAVG' in df['datatype'].values:
            tavg_data = df[df['datatype'] == 'TAVG']['value']
            avg_temp = tavg_data.mean()

            if params.optimal_temp_range[0] <= avg_temp <= params.optimal_temp_range[1]:
                suitability['temperature_suitable'] = True
            else:
                suitability['overall_score'] -= 30
                suitability['limiting_factors'].append('temperature')

        # Precipitation assessment
        if 'PRCP' in df['datatype'].values or 'TPCP' in df['datatype'].values:
            precip_var = 'PRCP' if 'PRCP' in df['datatype'].values else 'TPCP'
            precip_data = df[df['datatype'] == precip_var]['value']
            total_precip = precip_data.sum()

            if params.optimal_precip_range[0] <= total_precip <= params.optimal_precip_range[1]:
                suitability['precipitation_suitable'] = True
            else:
                suitability['overall_score'] -= 30
                suitability['limiting_factors'].append('precipitation')

        # Determine stress level
        if suitability['overall_score'] >= 80:
            suitability['stress_level'] = 'low'
        elif suitability['overall_score'] >= 60:
            suitability['stress_level'] = 'moderate'
        else:
            suitability['stress_level'] = 'high'

        return suitability


# =============================================================================
# Data Export and Conversion
# =============================================================================

class ModelDataConverter:
    """Convert climate data to formats for agricultural modeling tools."""

    @staticmethod
    def to_apsim_format(df: pd.DataFrame, station_id: str = None) -> pd.DataFrame:
        """
        Convert to APSIM weather file format.

        Args:
            df: Climate data DataFrame
            station_id: Optional station identifier

        Returns:
            DataFrame in APSIM format
        """
        # Pivot to get daily data
        daily_data = df.pivot_table(
            index=['station', 'date'], columns='datatype', values='value'
        ).reset_index()

        # Create APSIM columns
        apsim_df = pd.DataFrame()
        apsim_df['year'] = daily_data['date'].dt.year
        apsim_df['day'] = daily_data['date'].dt.dayofyear

        # Temperature (already in Celsius)
        if 'TMAX' in daily_data.columns:
            apsim_df['maxt'] = daily_data['TMAX']
        if 'TMIN' in daily_data.columns:
            apsim_df['mint'] = daily_data['TMIN']

        # Precipitation (already in mm)
        if 'PRCP' in daily_data.columns:
            apsim_df['rain'] = daily_data['PRCP']

        # Solar radiation (estimate if not available)
        if 'SRAD' in daily_data.columns:
            apsim_df['radn'] = daily_data['SRAD']
        else:
            # Simple estimation based on temperature range
            if 'TMAX' in daily_data.columns and 'TMIN' in daily_data.columns:
                temp_range = daily_data['TMAX'] - daily_data['TMIN']
                apsim_df['radn'] = 15 + (temp_range * 0.5)

        if station_id:
            apsim_df['station'] = station_id
        elif 'station' in daily_data.columns:
            apsim_df['station'] = daily_data['station']

        return apsim_df

    @staticmethod
    def to_dssat_format(df: pd.DataFrame) -> pd.DataFrame:
        """
        Convert to DSSAT weather file format.

        Args:
            df: Climate data DataFrame

        Returns:
            DataFrame in DSSAT format
        """
        # Similar to APSIM but with different column names
        daily_data = df.pivot_table(
            index=['station', 'date'], columns='datatype', values='value'
        ).reset_index()

        dssat_df = pd.DataFrame()
        dssat_df['YEAR'] = daily_data['date'].dt.year
        dssat_df['DOY'] = daily_data['date'].dt.dayofyear

        if 'TMAX' in daily_data.columns:
            dssat_df['TMAX'] = daily_data['TMAX']
        if 'TMIN' in daily_data.columns:
            dssat_df['TMIN'] = daily_data['TMIN']
        if 'PRCP' in daily_data.columns:
            dssat_df['RAIN'] = daily_data['PRCP']

        # Solar radiation
        if 'SRAD' in daily_data.columns:
            dssat_df['SRAD'] = daily_data['SRAD']
        else:
            if 'TMAX' in daily_data.columns and 'TMIN' in daily_data.columns:
                temp_range = daily_data['TMAX'] - daily_data['TMIN']
                dssat_df['SRAD'] = 15 + (temp_range * 0.5)

        return dssat_df

    @staticmethod
    def export_to_file(df: pd.DataFrame, filename: str, format_type: str = 'csv',
                       metadata: Dict = None):
        """
        Export data to file with proper headers.

        Args:
            df: Data to export
            filename: Output filename
            format_type: Format type ('csv', 'apsim', 'dssat')
            metadata: Optional metadata to include in header
        """
        with open(filename, 'w') as f:
            # Write header
            f.write(f"! Agricultural Climate Data Export\n")
            f.write(f"! Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"! Source: NOAA NCEI via agclimate module\n")

            if metadata:
                for key, value in metadata.items():
                    f.write(f"! {key}: {value}\n")

            f.write("!\n")

            # Write data
            df.to_csv(f, index=False)

        print(f"Data exported to {filename}")


# =============================================================================
# Utility Functions
# =============================================================================

def create_climate_report(analysis_results: Dict, output_file: str = None) -> str:
    """
    Create a formatted climate analysis report.

    Args:
        analysis_results: Results from CropClimateAnalyzer
        output_file: Optional file to save report

    Returns:
        Report as string
    """
    report = []
    report.append("=" * 60)
    report.append("AGRICULTURAL CLIMATE ANALYSIS REPORT")
    report.append("=" * 60)
    report.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report.append("")

    # Crop information
    crop_info = analysis_results.get('parameters')
    if crop_info:
        report.append(f"Crop: {crop_info.name}")
        report.append(f"Growing Season: Month {crop_info.start_month} to {crop_info.end_month}")
        report.append(
            f"Optimal Temperature Range: {crop_info.optimal_temp_range[0]:.1f} - {crop_info.optimal_temp_range[1]:.1f}°C")
        report.append(
            f"Optimal Precipitation: {crop_info.optimal_precip_range[0]:.0f} - {crop_info.optimal_precip_range[1]:.0f} mm")
        report.append("")

    # Data summary
    summary = analysis_results.get('data_summary', {})
    if summary:
        report.append("DATA SUMMARY")
        report.append("-" * 40)
        report.append(f"Total Records: {summary.get('total_records', 0):,}")
        report.append(f"Stations: {summary.get('stations', 0)}")
        report.append(f"Missing Data: {summary.get('missing_percentage', 0):.1f}%")
        if summary.get('avg_quality_score'):
            report.append(f"Average Quality Score: {summary['avg_quality_score']:.1f}/100")
        report.append("")

    # Growing season statistics
    season_stats = analysis_results.get('growing_season_stats')
    if season_stats is not None and not season_stats.empty:
        report.append("GROWING SEASON STATISTICS")
        report.append("-" * 40)

        # Temperature summary
        if 'TMAX_mean' in season_stats.columns:
            report.append(f"Average Max Temperature: {season_stats['TMAX_mean'].mean():.1f}°C")
        if 'TMIN_mean' in season_stats.columns:
            report.append(f"Average Min Temperature: {season_stats['TMIN_mean'].mean():.1f}°C")
        if 'precip_total' in season_stats.columns:
            report.append(f"Average Total Precipitation: {season_stats['precip_total'].mean():.0f} mm")
        report.append("")

    # GDD Analysis
    gdd_analysis = analysis_results.get('gdd_analysis')
    if gdd_analysis is not None and not gdd_analysis.empty:
        report.append("GROWING DEGREE DAYS ANALYSIS")
        report.append("-" * 40)
        annual_gdd = gdd_analysis.groupby('year')['gdd'].sum().mean()
        report.append(f"Average Annual GDD: {annual_gdd:.0f}")
        report.append("")

    # Stress indices
    stress = analysis_results.get('stress_indices', {})
    if stress:
        report.append("STRESS INDICES")
        report.append("-" * 40)
        if 'heat_stress_days' in stress:
            report.append(f"Heat Stress Days: {stress['heat_stress_days']}")
        if 'frost_days' in stress:
            report.append(f"Frost Days: {stress['frost_days']}")
        if 'max_dry_streak' in stress:
            report.append(f"Maximum Consecutive Dry Days: {stress['max_dry_streak']}")
        report.append("")

    # Suitability assessment
    suitability = analysis_results.get('suitability_assessment', {})
    if suitability:
        report.append("SUITABILITY ASSESSMENT")
        report.append("-" * 40)
        report.append(f"Overall Score: {suitability.get('overall_score', 0)}/100")
        report.append(f"Temperature Suitable: {suitability.get('temperature_suitable', False)}")
        report.append(f"Precipitation Suitable: {suitability.get('precipitation_suitable', False)}")
        report.append(f"Stress Level: {suitability.get('stress_level', 'unknown')}")

        limiting = suitability.get('limiting_factors', [])
        if limiting:
            report.append(f"Limiting Factors: {', '.join(limiting)}")
        report.append("")

    report.append("=" * 60)
    report.append("END OF REPORT")
    report.append("=" * 60)

    report_text = "\n".join(report)

    if output_file:
        with open(output_file, 'w') as f:
            f.write(report_text)
        print(f"Report saved to {output_file}")

    return report_text


# =============================================================================
# Example Usage
# =============================================================================

if __name__ == "__main__":
    # Example: Complete agricultural climate analysis workflow

    # Initialize API
    api = AgClimateAPI(token=os.getenv('NCEI_TOKEN'))

    # Example 1: Get daily data for Corn Belt
    print("Retrieving GHCND daily data for Corn Belt...")
    daily_data = api.get_agricultural_data(
        dataset='GHCND',
        region='corn_belt',
        start_date=date(2022, 1, 1),
        end_date=date(2023, 12, 31),
        variables=['TMIN', 'TMAX', 'PRCP'],
        max_stations=5
    )

    if not daily_data.empty:
        print(f"Retrieved {len(daily_data)} daily records")

        # Quality assessment
        processor = DataQualityProcessor()
        daily_data = processor.assess_quality(daily_data)
        daily_data = processor.fill_missing_values(daily_data, method='interpolation')

        # Analyze for corn
        analyzer = CropClimateAnalyzer()
        corn_analysis = analyzer.analyze_crop_climate(daily_data, 'corn')

        # Generate report
        report = create_climate_report(corn_analysis, 'corn_climate_report.txt')
        print("\nReport Preview:")
        print(report[:500] + "...")
