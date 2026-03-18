"""
Agricultural Climate Analysis Pipeline
=======================================
This module provides a complete pipeline for creating weather indices from USDA agricultural data:
1. Fetch county-level crop data (from CSV or NASS API)
2. Locate weather stations by agricultural density clustering
3. Fetch NCEI climate station data using AgClimateAPI
4. Generate weather indices for agricultural regions
5. Save/load station configurations for reuse

Author: Agricultural Data Analysis Team
Date: 2024
"""

import pandas as pd
import numpy as np
from datetime import date, datetime
from typing import Dict, List, Union
import warnings
from pathlib import Path

# Visualization libraries
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import json
import time
from urllib.request import urlopen

# Import existing modules
from .county_locator import clean_agricultural_data
from .agclimate import AgClimateAPI, CROP_PARAMETERS
from .config import NCEI_TOKEN
from MacrOSINT.data import NASSTable
from .density_station_locator import DensityBasedLocator

warnings.filterwarnings('ignore')
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")


class ClimatePipeline:
    """
    Complete pipeline for agricultural climate analysis from county data to weather indices.
    """

    def __init__(self, ncei_token: str = None):
        """
        Initialize the climate pipeline.
        
        Args:
            ncei_token: NCEI API token (uses config if not provided)
        """
        self.ncei_token = ncei_token or NCEI_TOKEN
        self.agclimate_api = AgClimateAPI() if self.ncei_token else None
        self.county_data = None
        self.top_counties = None
        # Removed grid_results - using agricultural density approach only
        self.climate_data = {}
        self.weather_indices = {}

    def load_county_data(self, source: Union[str, Dict], commodity: str = None, year: Union[str, int] = None,
                         **kwargs) -> pd.DataFrame:
        """
        Load county-level agricultural data from CSV file or NASS API.
        
        Args:
            source: Either CSV file path or 'nass' for API data
            **kwargs: Additional parameters for data loading
            
        Returns:
            Cleaned DataFrame with county agricultural data
            :param commodity:
            :param year:
        """
        if not commodity:
            commodity_desc = "CORN"
        else:
            commodity_desc = commodity.upper()
        if not year:
            year = "2024"
        else:
            if isinstance(year, int):
                year = str(year)
        if isinstance(source, str) and source.lower() == 'nass':
            return self._load_from_nass(nass_commodity=commodity_desc, year= year, **kwargs)
        if isinstance(source, str) and source.lower() == 'nass_api':
            return self._load_from_nass(nass_commodity=commodity_desc, year= year, use_api=True, **kwargs)
        elif isinstance(source, str):
            return self._load_from_csv(source)
        elif isinstance(source, dict):
            return self._load_from_nass_params(source)
        else:
            raise ValueError("Source must be CSV file path, 'nass', or NASS parameters dict")

    def _load_from_csv(self, filepath: str) -> pd.DataFrame:
        """Load data from CSV file using existing county_locator functions."""
        print(f"Loading county data from CSV: {filepath}")
        self.county_data = clean_agricultural_data(filepath)
        return self.county_data

    def _load_from_nass(self, nass_commodity='CORN', year="2024", use_api=False, **params) -> pd.DataFrame:
        """Load data from NASS API using comm_dash (if available)."""

            # Try to import and use comm_dash
        print("Loading county data from NASS API...")
        nass_client = NASSTable(commodity_desc=nass_commodity)

        # Default parameters for corn acres planted
        # Fetch data
        if not use_api:
            for key in nass_client.available_keys():
                # Searching for key in between start/enm
                if 'acres_planted' in key:
                    split_key =  key.split('/')
                    prod_key  = split_key[-1].split('_')
                    if len(prod_key) == 2:
                        start_year = int(prod_key[0])
                        end_year = int(prod_key[1])
                        selected_year = int(year)
                        if selected_year in {*range(start_year, end_year)}:
                            raw_df = nass_client[key]
                            raw_data = raw_df[raw_df['year'].astype(str) == year]
                            break
                    elif len(prod_key) == 1 and prod_key[0] == year:
                        raw_data = nass_client[key]
            else:
                # Fallback on API
                raw_data = nass_client._get_acres_planted(nass_commodity, year=year, update_keys=False)
        else:
            raw_data = nass_client._get_acres_planted(nass_commodity, year=year, update_keys=False)


        # Convert to DataFrame format matching CSV structure
        if isinstance(raw_data, pd.DataFrame):
            self.county_data = self._standardize_nass_data(raw_data)
        else:
            raise ValueError("NASS API returned unexpected data format")

        return self.county_data


    def _load_from_nass_params(self, params: Dict) -> pd.DataFrame:
        """Load data from NASS API using parameter dictionary."""
        return self._load_from_nass(**params)

    def _standardize_nass_data(self, nass_df: pd.DataFrame) -> pd.DataFrame:
        """Convert NASS API data to standardized format."""
        # This would need to be implemented based on actual NASS API response format
        # For now, assume the data is already in the correct format
        return clean_agricultural_data(nass_df)

    # Grid-based methods removed - now using agricultural density approach only

    def locate_stations_by_agricultural_density(self,
                                               clustering_method: str = 'weighted_kmeans',
                                               max_clusters: int = 12,
                                               min_acres: int = 50000,
                                               max_stations_per_cluster: int = 2,
                                               dataset: str = 'GHCND',
                                               start_date: date = None,
                                               end_date: date = None,
                                               min_coverage: float = 0.8) -> Dict[str, any]:
        """
        Locate weather stations based on agricultural activity density using clustering.
        
        This method identifies clusters of high agricultural activity and finds the
        optimal weather stations to represent each cluster, providing better coverage
        of agriculturally important areas than the grid-based approach.
        
        Args:
            clustering_method: Method for identifying clusters ('dbscan', 'kmeans', 'weighted_kmeans')
            max_clusters: Maximum number of clusters to create
            min_acres: Minimum acres for inclusion in clustering
            max_stations_per_cluster: Maximum stations to select per cluster
            dataset: NCEI dataset type ('GHCND' or 'GSOM')
            start_date: Start date for station coverage analysis
            end_date: End date for station coverage analysis
            min_coverage: Minimum data coverage threshold for stations
            
        Returns:
            Dictionary containing clusters, matched stations, and analysis results
        """
        if self.county_data is None:
            raise ValueError("County data not loaded. Call load_county_data() first.")
        
        print("Locating weather stations based on agricultural density clusters...")
        print("=" * 70)
        
        # Initialize the general density-based station locator
        locator = DensityBasedLocator(self.ncei_token)
        
        # Step 1: Prepare county data with coordinates
        print(f"Step 1: Preparing county data with geocoding...")
        county_data_with_coords = self.county_data.copy()
        
        # Add coordinates if not already present
        if 'latitude' not in county_data_with_coords.columns or 'longitude' not in county_data_with_coords.columns:
            from .add_county_centroids import append_centroids_via_shapes as append_centroids
            county_data_with_coords = append_centroids(county_data_with_coords)
            county_data_with_coords = county_data_with_coords.dropna(subset=['latitude', 'longitude'])
            print(f"  Added coordinates to {len(county_data_with_coords)} counties")
        
        # Ensure required columns are present
        if 'acres_planted' not in county_data_with_coords.columns and 'Value' in county_data_with_coords.columns:
            county_data_with_coords['acres_planted'] = county_data_with_coords['Value']
        
        # Step 2: Identify agricultural clusters using general density approach
        print(f"\nStep 2: Identifying agricultural clusters...")
        clusters = locator.identify_density_clusters(
            data=county_data_with_coords,
            density_column='acres_planted',
            county_column='county_name',
            lat_column='latitude',
            lon_column='longitude',
            fips_column='fips' if 'fips' in county_data_with_coords.columns else None,
            method=clustering_method,
            min_density_threshold=min_acres,
            max_clusters=max_clusters
        )
        
        if not clusters:
            print("WARNING: No agricultural clusters identified")
            return {
                'clusters': [],
                'stations': [],
                'analysis_summary': 'No clusters found'
            }
        
        # Step 3: Find optimal weather stations for each cluster
        print(f"\nStep 3: Finding optimal weather stations...")
        stations = locator.find_optimal_stations(
            clusters=clusters,
            dataset=dataset,
            start_date=start_date,
            end_date=end_date,
            min_coverage=min_coverage,
            max_stations_per_cluster=max_stations_per_cluster
        )
        
        # Step 4: Export results
        print(f"\nStep 4: Exporting results...")
        exported_files = locator.export_results(clusters, stations, 'agricultural_cluster_analysis')
        
        # Convert clusters to dictionary format for compatibility
        clusters_dict = []
        for cluster in clusters:
            clusters_dict.append({
                'cluster_id': cluster.cluster_id,
                'center_latitude': cluster.center_lat,
                'center_longitude': cluster.center_lon,
                'total_acres': cluster.total_density,  # total_density is acres for agricultural data
                'county_count': cluster.county_count,
                'counties': cluster.counties,
                'radius_km': cluster.radius_km,
                'density_score': cluster.density_score,
                'fips_codes': cluster.fips_codes
            })
        
        # Convert stations to dictionary format for compatibility
        stations_dict = []
        for station in stations:
            stations_dict.append({
                'station_id': station.station_id,
                'station_name': station.station_name,
                'latitude': station.latitude,
                'longitude': station.longitude,
                'cluster_id': station.cluster_id,
                'distance_km': station.distance_km,
                'data_coverage': station.data_coverage,
                'match_score': station.match_score
            })
        
        # Store results in pipeline object (both dict and object formats for compatibility)
        self.agricultural_clusters = clusters_dict
        self.weather_stations = stations_dict
        self.agricultural_stations = stations_dict
        
        results = {
            'clusters': clusters_dict,
            'stations': stations_dict,
            'analysis_summary': f"Agricultural density analysis completed with {len(clusters)} clusters",
            'exported_files': exported_files,
            'total_agricultural_area': sum(c.total_density for c in clusters),
            'total_counties_represented': sum(c.county_count for c in clusters),
            'stations_found': len(stations),
            'average_station_coverage': np.mean([s.data_coverage for s in stations]) if stations else 0,
            'average_distance_to_cluster': np.mean([s.distance_km for s in stations]) if stations else 0
        }
        
        print(f"\nAgricultural Cluster Analysis Complete!")
        print(f"  Clusters identified: {len(clusters)}")
        print(f"  Stations found: {len(stations)}")
        print(f"  Total agricultural area: {results['total_agricultural_area']:,.0f} acres")
        print(f"  Average station coverage: {results['average_station_coverage']:.1%}")
        print(f"  Average distance to cluster: {results['average_distance_to_cluster']:.1f} km")
        
        return results

    def fetch_climate_data_from_agricultural_stations(self,
                                                      station_analysis: Dict,
                                                      start_date: date,
                                                      end_date: date,
                                                      dataset: str = 'GHCND',
                                                      variables: List[str] = None) -> Dict[str, pd.DataFrame]:
        """
        Fetch climate data using the agriculturally-optimized station selection.
        
        Args:
            agricultural_analysis: Results from locate_stations_by_agricultural_density()
            start_date: Start date for climate data
            end_date: End date for climate data
            dataset: NCEI dataset type
            variables: Climate variables to fetch
            
        Returns:
            Dictionary mapping cluster IDs to climate DataFrames
        """
        if not agricultural_analysis['stations']:
            print("WARNING: No stations available from agricultural analysis")
            return {}
        
        print(f"Fetching climate data from {len(agricultural_analysis['stations'])} agriculturally-optimized stations...")
        
        # Default variables for agricultural analysis
        if variables is None:
            variables = ['TMAX', 'TMIN', 'PRCP', 'TOBS'] if dataset == 'GHCND' else ['TMAX', 'TMIN', 'PRCP']
        
        climate_data = {}
        successful_stations = 0
        failed_stations = 0
        
        # Group stations by cluster
        stations_by_cluster = {}
        for station in agricultural_analysis['stations']:
            cluster_id = station['cluster_id']  # Now accessing dictionary key
            if cluster_id not in stations_by_cluster:
                stations_by_cluster[cluster_id] = []
            stations_by_cluster[cluster_id].append(station)
        
        # Fetch data for each cluster
        for cluster_id, cluster_stations in stations_by_cluster.items():
            cluster = next(c for c in agricultural_analysis['clusters'] if c['cluster_id'] == cluster_id)
            print(f"\nFetching data for Cluster {cluster_id} ({cluster['total_acres']:,.0f} acres)...")
            
            cluster_data = []
            
            for station in cluster_stations:
                print(f"  Processing station {station['station_name']} ({station['station_id']})...")
                
                try:
                    # Fetch data year by year for GHCND (due to API limitations)
                    if dataset == 'GHCND':
                        station_data = self._fetch_ghcnd_station_data(
                            station['station_id'], station['station_name'], 
                            start_date, end_date, variables,
                            station['latitude'], station['longitude']
                        )
                    else:
                        station_data = self._fetch_gsom_station_data(
                            station['station_id'], station['station_name'],
                            start_date, end_date, variables,
                            station['latitude'], station['longitude']
                        )
                    
                    if not station_data.empty:
                        cluster_data.append(station_data)
                        successful_stations += 1
                        print(f"    SUCCESS: Retrieved {len(station_data)} records")
                    else:
                        print(f"    WARNING: No data retrieved for {station['station_id']}")
                        failed_stations += 1
                
                except Exception as e:
                    print(f"    ERROR: Failed to retrieve data for {station['station_id']}: {e}")
                    failed_stations += 1
                    continue
            
            # Combine data for this cluster
            if cluster_data:
                combined_cluster_data = pd.concat(cluster_data, ignore_index=True)
                processed_data = self._process_ncei_data(combined_cluster_data, dataset)
                
                if not processed_data.empty:
                    climate_data[f"cluster_{cluster_id}"] = processed_data
                    print(f"  SUCCESS: Cluster {cluster_id} has {len(processed_data)} processed records")
        
        print(f"\nClimate Data Retrieval Summary:")
        print(f"  Total clusters processed: {len(stations_by_cluster)}")
        print(f"  Successful stations: {successful_stations}")
        print(f"  Failed stations: {failed_stations}")
        print(f"  Clusters with data: {len(climate_data)}")
        
        # Store in pipeline object
        self.climate_data = climate_data
        
        return climate_data

    def _fetch_ghcnd_station_data(self, station_id: str, station_name: str,
                                start_date: date, end_date: date, variables: List[str],
                                latitude: float, longitude: float) -> pd.DataFrame:
        """Fetch GHCND data for a single station with year-by-year chunking."""
        all_data = []
        
        current_year = start_date.year
        while current_year <= end_date.year:
            year_start = max(start_date, date(current_year, 1, 1))
            year_end = min(end_date, date(current_year, 12, 31))
            
            try:
                year_data = self.agclimate_api.ncei.get_data(
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
                        df['station_name'] = station_name
                        df['latitude'] = latitude
                        df['longitude'] = longitude
                        all_data.append(df)
                
                time.sleep(0.2)  # Rate limiting
                
            except Exception as e:
                print(f"    Error retrieving {current_year} data: {e}")
            
            current_year += 1
        
        return pd.concat(all_data, ignore_index=True) if all_data else pd.DataFrame()

    def _fetch_gsom_station_data(self, station_id: str, station_name: str,
                               start_date: date, end_date: date, variables: List[str],
                               latitude: float, longitude: float) -> pd.DataFrame:
        """Fetch GSOM data for a single station."""
        try:
            monthly_data = self.agclimate_api.ncei.get_data(
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
                    df['station_name'] = station_name
                    df['latitude'] = latitude
                    df['longitude'] = longitude
                    return df
            
            time.sleep(0.2)  # Rate limiting
            
        except Exception as e:
            print(f"    Error retrieving GSOM data: {e}")
        
        return pd.DataFrame()

    def _create_agricultural_extents(self, agricultural_analysis: Dict) -> Dict[str, str]:
        """
        Create extent strings for agricultural clusters for compatibility with existing functions.
        
        Args:
            agricultural_analysis: Results from locate_stations_by_agricultural_density()
            
        Returns:
            Dictionary mapping cluster names to extent strings
        """
        extents = {}
        
        if not agricultural_analysis.get('clusters'):
            return extents
        
        for cluster in agricultural_analysis['clusters']:
            # Create extent string with some padding around cluster center
            padding = cluster['radius_km'] / 111  # Convert km to degrees (approximate)
            min_lat = cluster['center_latitude'] - padding
            max_lat = cluster['center_latitude'] + padding
            min_lon = cluster['center_longitude'] - padding
            max_lon = cluster['center_longitude'] + padding
            
            extent_str = f"{min_lat:.2f},{min_lon:.2f},{max_lat:.2f},{max_lon:.2f}"
            cluster_name = f"cluster_{cluster['cluster_id']}"
            extents[cluster_name] = extent_str
            
            print(f"  {cluster_name}: {extent_str} (Agricultural: {cluster['total_acres']:,.0f} acres)")
        
        return extents

    # Grid-based fetch_climate_data method removed - using fetch_climate_data_from_agricultural_stations instead
    
    def _process_ncei_data(self, data: pd.DataFrame, dataset: str = 'GHCND') -> pd.DataFrame:
        """
        Process NCEI data from long format to wide format with proper unit conversions.
        
        Args:
            data: Raw NCEI data in long format
            dataset: Dataset type for unit conversion
            
        Returns:
            Processed DataFrame in wide format with proper units
        """
        if data.empty:
            print(f"    WARNING: Input data is empty")
            return pd.DataFrame()
        
        try:
            # Apply unit conversions first
            data_converted = data.copy()
            
            if dataset == 'GHCND':
                # Keep temperatures in Celsius (crop parameters expect Celsius)
                # GHCND temperatures are already in degrees Celsius (API pre-converts from tenths)
                
                # Keep precipitation in mm (crop parameters work with mm)
                # GHCND precipitation is already in mm (API pre-converts from tenths)
                pass
            
            # Pivot from long format to wide format
            # Keep essential columns and pivot on datatype
            essential_cols = ['station', 'date', 'latitude', 'longitude', 'elevation']
            available_cols = [col for col in essential_cols if col in data_converted.columns]
            
            if not available_cols:
                print(f"    ERROR: No essential columns found in data. Available columns: {list(data_converted.columns)}")
                return pd.DataFrame()
            
            # Check for required columns
            if 'datatype' not in data_converted.columns:
                print(f"    ERROR: 'datatype' column missing from data")
                return pd.DataFrame()
            
            if 'value' not in data_converted.columns:
                print(f"    ERROR: 'value' column missing from data")
                return pd.DataFrame()
            
            # Create pivot table
            wide_data = data_converted.pivot_table(
                index=available_cols,
                columns='datatype',
                values='value',
                aggfunc='mean'  # Average if multiple readings per day
            ).reset_index()
            
            # Clean up column names (remove the name from columns index)
            wide_data.columns.name = None
            
            # Filter out any completely NaN rows
            data_cols = [col for col in wide_data.columns if col not in available_cols]
            if data_cols:
                wide_data = wide_data.dropna(how='all', subset=data_cols)
            
            print(f"    Converted from long format ({len(data)} rows) to wide format ({len(wide_data)} rows)")
            if not wide_data.empty:
                temp_cols = [col for col in ['TMAX', 'TMIN', 'TAVG'] if col in wide_data.columns]
                precip_cols = [col for col in ['PRCP'] if col in wide_data.columns]
                print(f"    Available variables: {temp_cols + precip_cols}")
                
                # Show sample values (keeping original units)
                if 'TMAX' in wide_data.columns:
                    sample_tmax = wide_data['TMAX'].dropna().head(3)
                    if len(sample_tmax) > 0:
                        print(f"    Sample TMAX (°C): {list(sample_tmax.round(1))}")
                if 'PRCP' in wide_data.columns:
                    sample_prcp = wide_data['PRCP'].dropna().head(3)
                    if len(sample_prcp) > 0:
                        print(f"    Sample PRCP (mm): {list(sample_prcp.round(1))}")
            else:
                print(f"    WARNING: Wide format data is empty after processing")
            
            return wide_data
            
        except Exception as e:
            print(f"    ERROR: Failed to process NCEI data: {str(e)}")
            print(f"    Data shape: {data.shape if hasattr(data, 'shape') else 'unknown'}")
            print(f"    Data columns: {list(data.columns) if hasattr(data, 'columns') else 'unknown'}")
            return pd.DataFrame()

    def calculate_weather_indices(self, indices: List[str] = None, commodity: str = None) -> Dict[str, pd.DataFrame]:
        """
        Calculate weather indices for each agricultural cluster.
        
        Args:
            indices: List of indices to calculate
            commodity: Crop commodity for specialized parameters
            
        Returns:
            Dictionary mapping cluster names to weather index DataFrames
        """
        if not self.climate_data:
            print("WARNING: No climate data available. Skipping weather indices calculation.")
            return {}

        if indices is None:
            indices = ['gdd', 'precipitation_total', 'temperature_avg', 'stress_days']

        print("Calculating weather indices...")

        weather_indices = {}
        successful_calculations = 0
        failed_calculations = 0

        # Determine crop type for GDD calculation
        crop_type = None
        if commodity:
            commodity_lower = commodity.lower()
            if 'corn' in commodity_lower or 'maize' in commodity_lower:
                crop_type = 'corn'
            elif 'soybean' in commodity_lower or 'soya' in commodity_lower:
                crop_type = 'soybeans'
            elif 'wheat' in commodity_lower:
                crop_type = 'wheat_winter'  # Default to winter wheat
            elif 'cotton' in commodity_lower:
                crop_type = 'cotton'
            elif 'rice' in commodity_lower:
                crop_type = 'rice'

        for cluster_name, climate_df in self.climate_data.items():
            print(f"\nProcessing {cluster_name}...")

            try:
                indices_df = self._calculate_indices_for_cluster(climate_df, indices, crop_type)
                if not indices_df.empty:
                    weather_indices[cluster_name] = indices_df
                    successful_calculations += 1
                    print(f"  SUCCESS: Calculated {len(indices)} indices for {len(indices_df)} time periods")
                else:
                    print(f"  WARNING: No indices calculated for {cluster_name} (empty result)")
                    failed_calculations += 1
            except Exception as e:
                print(f"  ERROR: Failed to calculate indices for {cluster_name}: {str(e)}")
                failed_calculations += 1
                continue

        print(f"\nWeather Indices Summary:")
        print(f"  Successful: {successful_calculations}")
        print(f"  Failed: {failed_calculations}")
        
        self.weather_indices = weather_indices
        return weather_indices

    def _calculate_indices_for_cluster(self, climate_df: pd.DataFrame,
                                    indices: List[str], crop_type: str = None) -> pd.DataFrame:
        """Calculate weather indices for a single agricultural cluster."""
        
        if climate_df.empty:
            print(f"    WARNING: Empty climate dataframe provided")
            return pd.DataFrame()

        # Ensure date column exists
        try:
            if 'date' not in climate_df.columns:
                print(f"    ERROR: No 'date' column found in climate data")
                return pd.DataFrame()
            climate_df['date'] = pd.to_datetime(climate_df['date'])
        except Exception as e:
            print(f"    ERROR: Failed to process date column: {str(e)}")
            return pd.DataFrame()

        # Group by time period (monthly for now)
        climate_df['year_month'] = climate_df['date'].dt.to_period('M')
        monthly_stats = []

        # Get crop parameters for GDD calculation (keep in Celsius)
        if crop_type and crop_type in CROP_PARAMETERS:
            crop_params = CROP_PARAMETERS[crop_type]
            base_temp_c = crop_params.base_temp
            max_temp_c = crop_params.max_temp
            print(f"    Using {crop_params.name} parameters - Base temp: {base_temp_c:.1f}°C, Max temp: {max_temp_c:.1f}°C")
        else:
            # Default corn parameters if no crop specified
            base_temp_c = 10.0  # Corn base temperature in Celsius
            max_temp_c = 30.0   # Corn max temperature in Celsius
            if crop_type:
                print(f"    Unknown crop type '{crop_type}', using default corn parameters")

        for period, group in climate_df.groupby('year_month'):
            stats = {'period': str(period)}

            if 'gdd' in indices:
                # Growing Degree Days with proper crop parameters (Celsius)
                if 'TMAX' in group.columns and 'TMIN' in group.columns:
                    # Apply temperature caps (max temp threshold)
                    tmax_capped = np.minimum(group['TMAX'], max_temp_c)
                    tmin_capped = np.maximum(group['TMIN'], base_temp_c)
                    daily_avg_capped = (tmax_capped + tmin_capped) / 2
                    
                    # Calculate GDD with base temperature
                    daily_gdd = np.maximum(0, daily_avg_capped - base_temp_c)
                    stats['gdd'] = daily_gdd.sum()
                else:
                    stats['gdd'] = np.nan

            if 'precipitation_total' in indices:
                if 'PRCP' in group.columns:
                    stats['precipitation_total'] = group['PRCP'].sum()
                else:
                    stats['precipitation_total'] = np.nan

            if 'temperature_avg' in indices:
                if 'TMAX' in group.columns and 'TMIN' in group.columns:
                    stats['temperature_avg'] = (group['TMAX'] + group['TMIN']).mean() / 2
                else:
                    stats['temperature_avg'] = np.nan

            if 'stress_days' in indices:
                # Days with temperature above crop stress threshold
                if 'TMAX' in group.columns:
                    # Use crop-specific stress threshold, default to 32°C (90°F equivalent)
                    stress_threshold = max_temp_c if crop_type and crop_type in CROP_PARAMETERS else 32.0
                    stats['stress_days'] = (group['TMAX'] > stress_threshold).sum()
                else:
                    stats['stress_days'] = np.nan

            monthly_stats.append(stats)

        return pd.DataFrame(monthly_stats)

    def create_climate_visualizations(self,county_data, output_dir: str = "visuals", show_plots: bool = False):
        """Create comprehensive visualizations of climate data and analyses."""
        if not self.climate_data:
            print("WARNING: No climate data available for visualization")
            return

        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)

        print(f"Creating climate visualizations in {output_path}...")

        # 1. Station Coverage Map
        self._create_station_coverage_map(county_data, output_path, show_plots)

        # 2. Time Series Plots
        self._create_time_series_plots(output_path, show_plots)

        # 3. Weather Indices Heatmaps
        if self.weather_indices:
            self._create_weather_indices_heatmaps(output_path, show_plots)

        # 4. Grid Cell Comparison
        self._create_grid_comparison_plots(output_path, show_plots)

        # 5. Interactive Dashboard
        self._create_interactive_dashboard(output_path)

        print(f"SUCCESS: All visualizations saved to {output_path}")

    def _create_station_coverage_map(self, acreage_data : pd.DataFrame, output_path: Path, show_plots: bool):
        """Create a choropleth map showing planted acreage density with climate station overlays."""
        try:
            # Use fallback approach by default (as requested)
            print("    Using fallback approach for county boundaries...")
            
            # Load county planted acreage data
            if not acreage_data.empty:
                county_df = acreage_data.copy()
                
                # Load county boundaries from online geojson (fallback by default)
                print("    Loading county boundaries from online geojson...")
                with urlopen(
                        'https://raw.githubusercontent.com/plotly/datasets/master/geojson-counties-fips.json') as response:
                    counties = json.load(response)

                # Clean the Value column (planted acreage)
                def clean_acres(value):
                    if pd.isna(value):
                        return 0
                    value_str = str(value).replace(',', '').replace('"', '').strip()
                    try:
                        return int(value_str)
                    except (ValueError, TypeError):
                        return 0

                if "Value" in county_df.columns:
                    county_df['acres_planted'] = county_df['Value'].apply(clean_acres)
                elif 'acres_planted' in county_df.columns:
                    if county_df['acres_planted'].dtype == str:
                        county_df['acres_planted'] = county_df['acres_planted'].apply(clean_acres)
                    else:
                        county_df = county_df[county_df['acres_planted'] > 0]
                else:
                    print("Failed to locate acres_planted in columns")
                    raise KeyError

                # Create FIPS code for choropleth mapping
                county_df['fips'] = county_df['state_fips_code'].astype(str).str.zfill(2) + \
                                   county_df['county_code'].astype(str).str.zfill(3)

                # Create choropleth map using plotly express
                fig = px.choropleth(
                    county_df,
                    geojson=counties,
                    locations='fips',
                    color='acres_planted',
                    hover_name='county_name',
                    hover_data={'state_name': True, 'acres_planted': ':,'},
                    color_continuous_scale='YlOrRd',
                    scope="usa",
                    title="Agricultural Density with Climate Station Coverage (2024)",
                    labels={'acres_planted': 'Acres Planted'},
                    range_color=[0, county_df['acres_planted'].quantile(0.95)]
                )

                # Add agricultural clusters as scatter points (if available)
                if hasattr(self, 'agricultural_clusters') and self.agricultural_clusters:
                    colors = px.colors.qualitative.Set3[:len(self.agricultural_clusters)]
                    
                    for i, cluster in enumerate(self.agricultural_clusters):
                        fig.add_trace(go.Scattergeo(
                            lon=[cluster['center_longitude']],
                            lat=[cluster['center_latitude']],
                            mode='markers',
                            marker=dict(
                                size=20,
                                color=colors[i % len(colors)],
                                symbol='diamond',
                                line=dict(width=3, color='black')
                            ),
                            name=f'Cluster {cluster["cluster_id"]} ({cluster["total_acres"]:,.0f} acres)',
                            hovertemplate="<b>Agricultural Cluster %{text}</b><br>" +
                                          f"Total Acres: {cluster['total_acres']:,.0f}<br>" +
                                          f"Counties: {cluster['county_count']}<br>" +
                                          "Lat: %{lat:.3f}<br>" +
                                          "Lon: %{lon:.3f}<br>" +
                                          "<extra></extra>",
                            text=[f"{cluster['cluster_id']}"]
                        ))

                # Add climate stations as scatter points (if available)
                if hasattr(self, 'weather_stations') and self.weather_stations:
                    station_colors = px.colors.qualitative.Pastel[:len(self.weather_stations)]
                    
                    for i, station in enumerate(self.weather_stations):
                        cluster_color = colors[station['cluster_id'] % len(colors)] if hasattr(self, 'agricultural_clusters') and self.agricultural_clusters else 'red'
                        
                        fig.add_trace(go.Scattergeo(
                            lon=[station['longitude']],
                            lat=[station['latitude']],
                            mode='markers',
                            marker=dict(
                                size=12,
                                color=cluster_color,
                                symbol='circle',
                                line=dict(width=2, color='white')
                            ),
                            name=f'Station: {station["station_name"]}',
                            hovertemplate="<b>%{text}</b><br>" +
                                          f"Station ID: {station['station_id']}<br>" +
                                          f"Coverage: {station['data_coverage']:.1%}<br>" +
                                          f"Distance: {station['distance_km']:.1f} km<br>" +
                                          "Lat: %{lat:.3f}<br>" +
                                          "Lon: %{lon:.3f}<br>" +
                                          "<extra></extra>",
                            text=[station['station_name']]
                        ))

                # Update layout
                fig.update_layout(
                    title={
                        'text': 'Agricultural Density with Climate Station Coverage (2024)',
                        'x': 0.5,
                        'xanchor': 'center',
                        'font': {'size': 16}
                    },
                    geo=dict(
                        scope='usa',
                        projection_type='albers usa',
                        showlakes=True,
                        lakecolor='rgb(255, 255, 255)',
                    ),
                    height=700,
                    width=1200
                )

                # Save as HTML
                html_path = output_path / 'station_coverage_choropleth.html'
                fig.write_html(html_path)
                print(f"    Choropleth map saved: {html_path}")

                if show_plots:
                    fig.show()

            else:
                print(f"    County data is empty")
                
        except Exception as e:
            print(f"    Error creating choropleth map: {str(e)}")
            print("    Falling back to basic scatter plot...")
            
            # Fallback to basic matplotlib plot
            fig, ax = plt.subplots(figsize=(12, 8))
            
            # Plot agricultural clusters if available
            if hasattr(self, 'agricultural_clusters') and self.agricultural_clusters:
                cluster_lats = [c['center_latitude'] for c in self.agricultural_clusters]
                cluster_lons = [c['center_longitude'] for c in self.agricultural_clusters]
                cluster_sizes = [c['total_acres'] / 10000 for c in self.agricultural_clusters]  # Scale for visibility
                
                ax.scatter(cluster_lons, cluster_lats, s=cluster_sizes, alpha=0.6, 
                          c='red', marker='D', label='Agricultural Clusters')
            
            # Plot weather stations if available
            if hasattr(self, 'weather_stations') and self.weather_stations:
                station_lats = [s['latitude'] for s in self.weather_stations]
                station_lons = [s['longitude'] for s in self.weather_stations]
                
                ax.scatter(station_lons, station_lats, s=100, alpha=0.8,
                          c='blue', marker='o', label='Weather Stations')

            ax.set_xlabel('Longitude')
            ax.set_ylabel('Latitude')
            ax.set_title('Agricultural Clusters and Climate Stations Coverage')
            ax.legend()
            ax.grid(True, alpha=0.3)

            plt.tight_layout()
            plt.savefig(output_path / 'station_coverage_map.png', dpi=300, bbox_inches='tight')
            if show_plots:
                plt.show()
            plt.close()

    def _create_time_series_plots(self, output_path: Path, show_plots: bool):
        """Create time series plots for key climate variables."""
        variables = ['TMAX', 'TMIN', 'PRCP']

        for var in variables:
            fig, axes = plt.subplots(len(self.climate_data), 1,
                                     figsize=(15, 4 * len(self.climate_data)),
                                     sharex=True)

            if len(self.climate_data) == 1:
                axes = [axes]

            for i, (grid_cell, climate_df) in enumerate(self.climate_data.items()):
                if var in climate_df.columns:
                    # Convert date column and resample to monthly averages
                    df_copy = climate_df.copy()
                    df_copy['date'] = pd.to_datetime(df_copy['date'])
                    monthly_data = df_copy.groupby(df_copy['date'].dt.to_period('M'))[var].mean()

                    axes[i].plot(monthly_data.index.to_timestamp(), monthly_data.values,
                                 linewidth=2, color='blue')
                    axes[i].set_title(f'{var} - {grid_cell}')
                    axes[i].set_ylabel(f'{var} (°F)' if 'T' in var else f'{var} (inches)')
                    axes[i].grid(True, alpha=0.3)

            plt.xlabel('Date')
            plt.suptitle(f'{var} Time Series by Grid Cell', fontsize=16)
            plt.tight_layout()
            plt.savefig(output_path / f'{var.lower()}_time_series.png', dpi=300, bbox_inches='tight')
            if show_plots:
                plt.show()
            plt.close()

    def _create_weather_indices_heatmaps(self, output_path: Path, show_plots: bool):

        """Create heatmaps for weather indices across grid cells and time periods."""
        if not self.weather_indices:
            return

        # Combine all indices data
        all_indices = []
        for grid_cell, indices_df in self.weather_indices.items():
            df_copy = indices_df.copy()
            df_copy['grid_cell'] = grid_cell
            all_indices.append(df_copy)

        combined_df = pd.concat(all_indices, ignore_index=True)

        # Create heatmaps for each index
        indices_to_plot = ['gdd', 'precipitation_total', 'temperature_avg', 'stress_days']

        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        axes = axes.flatten()

        for i, index_name in enumerate(indices_to_plot):
            if index_name in combined_df.columns:
                pivot_data = combined_df.pivot(index='grid_cell', columns='period', values=index_name)

                sns.heatmap(pivot_data, annot=True, fmt='.1f', cmap='viridis',
                            ax=axes[i], cbar_kws={'label': index_name})
                axes[i].set_title(f'{index_name.replace("_", " ").title()}')
                axes[i].set_xlabel('Time Period')
                axes[i].set_ylabel('Grid Cell')

        plt.tight_layout()
        plt.savefig(output_path / 'weather_indices_heatmaps.png', dpi=300, bbox_inches='tight')
        if show_plots:
            plt.show()
        plt.close()

    def _create_grid_comparison_plots(self, output_path: Path, show_plots: bool):
        """Create comparison plots between grid cells."""
        if len(self.climate_data) < 2:
            return

        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        variables = ['TMAX', 'TMIN', 'PRCP']

        # Box plots comparing variables across grid cells
        for i, var in enumerate(variables):
            if i < 3:  # Only plot first 3 variables
                row, col = divmod(i, 2)

                data_for_plot = []
                labels = []

                for grid_cell, climate_df in self.climate_data.items():
                    if var in climate_df.columns:
                        data_for_plot.append(climate_df[var].dropna())
                        labels.append(grid_cell)

                if data_for_plot:
                    axes[row, col].boxplot(data_for_plot, labels=labels)
                    axes[row, col].set_title(f'{var} Distribution by Grid Cell')
                    axes[row, col].set_ylabel(f'{var} (°F)' if 'T' in var else f'{var} (inches)')
                    axes[row, col].tick_params(axis='x', rotation=45)

        # Remove empty subplot
        if len(variables) == 3:
            fig.delaxes(axes[1, 1])

        plt.tight_layout()
        plt.savefig(output_path / 'grid_comparison_boxplots.png', dpi=300, bbox_inches='tight')
        if show_plots:
            plt.show()
        plt.close()

    def _create_interactive_dashboard(self, output_path: Path):
        """Create an interactive Plotly dashboard."""
        if not self.climate_data:
            return

        # Prepare data for interactive plots
        all_data = []
        for grid_cell, climate_df in self.climate_data.items():
            df_copy = climate_df.copy()
            df_copy['grid_cell'] = grid_cell
            all_data.append(df_copy)

        combined_df = pd.concat(all_data, ignore_index=True)
        combined_df['date'] = pd.to_datetime(combined_df['date'])

        # Create subplots - separate map from other plots
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Temperature Max', 'Temperature Min', 'Precipitation', 'Station Locations'),
            specs=[[{"secondary_y": False}, {"secondary_y": False}],
                   [{"secondary_y": False}, {"type": "scatter"}]]
        )

        # Add temperature plots
        for grid_cell in combined_df['grid_cell'].unique():
            grid_data = combined_df[combined_df['grid_cell'] == grid_cell]

            if 'TMAX' in grid_data.columns:
                fig.add_trace(
                    go.Scatter(x=grid_data['date'], y=grid_data['TMAX'],
                               name=f'TMAX - {grid_cell}', mode='lines'),
                    row=1, col=1
                )

            if 'TMIN' in grid_data.columns:
                fig.add_trace(
                    go.Scatter(x=grid_data['date'], y=grid_data['TMIN'],
                               name=f'TMIN - {grid_cell}', mode='lines'),
                    row=1, col=2
                )

            if 'PRCP' in grid_data.columns:
                fig.add_trace(
                    go.Scatter(x=grid_data['date'], y=grid_data['PRCP'],
                               name=f'PRCP - {grid_cell}', mode='lines'),
                    row=2, col=1
                )

        # Add station locations plot (regular scatter)
        if 'latitude' in combined_df.columns and 'longitude' in combined_df.columns:
            stations = combined_df.groupby(['station', 'grid_cell']).agg({
                'latitude': 'first',
                'longitude': 'first'
            }).reset_index()

            fig.add_trace(
                go.Scatter(
                    x=stations['longitude'],
                    y=stations['latitude'],
                    mode='markers',
                    marker=dict(size=10),
                    text=stations['station'],
                    name='Stations'
                ),
                row=2, col=2
            )

        fig.update_layout(
            title="Climate Data Interactive Dashboard",
            height=800,
            showlegend=True
        )

        # Update axis labels for station plot
        fig.update_xaxes(title_text="Longitude", row=2, col=2)
        fig.update_yaxes(title_text="Latitude", row=2, col=2)

        # Save interactive plot
        fig.write_html(str(output_path / 'interactive_dashboard.html'))
        print(f"SUCCESS: Interactive dashboard saved: {output_path / 'interactive_dashboard.html'}")

    def export_results(self, output_dir: str = ".", prefix: str = "climate_pipeline"):
        """
        Export all pipeline results to files.
        
        Args:
            output_dir: Output directory
            prefix: File prefix
        """
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)

        print(f"Exporting results to {output_path}...")

        # Export county data
        if self.county_data is not None:
            filepath = output_path / f"{prefix}_county_data.csv"
            self.county_data.to_csv(filepath, index=False)
            print(f"  SUCCESS: County data: {filepath}")

        # Export top counties
        if self.top_counties is not None:
            filepath = output_path / f"{prefix}_top_counties.csv"
            self.top_counties.to_csv(filepath, index=False)
            print(f"  SUCCESS: Top counties: {filepath}")

        # Export grid results
        if self.grid_results is not None:
            filepath = output_path / f"{prefix}_grid_analysis.csv"
            self.grid_results['high_density_counties'].to_csv(filepath, index=False)
            print(f"  SUCCESS: Grid analysis: {filepath}")

        # Export combined climate data (all grid cells in one file)
        if self.climate_data:
            all_climate_data = []
            for grid_cell, data in self.climate_data.items():
                data_with_grid = data.copy()
                data_with_grid['grid_cell'] = grid_cell
                all_climate_data.append(data_with_grid)
            
            if all_climate_data:
                combined_climate = pd.concat(all_climate_data, ignore_index=True)
                filepath = output_path / f"{prefix}_climate_data.csv"
                combined_climate.to_csv(filepath, index=False)
                print(f"  SUCCESS: Combined climate data ({len(self.climate_data)} grid cells): {filepath}")
        
        # Export combined weather indices (all grid cells in one file)
        if self.weather_indices:
            all_indices_data = []
            for grid_cell, indices_df in self.weather_indices.items():
                indices_with_grid = indices_df.copy()
                indices_with_grid['grid_cell'] = grid_cell
                all_indices_data.append(indices_with_grid)
            
            if all_indices_data:
                combined_indices = pd.concat(all_indices_data, ignore_index=True)
                filepath = output_path / f"{prefix}_weather_indices.csv"
                combined_indices.to_csv(filepath, index=False)
                print(f"  SUCCESS: Combined weather indices ({len(self.weather_indices)} grid cells): {filepath}")

    def run_complete_pipeline(self, source: Union[str, Dict],
                              start_date: date, end_date: date,
                              commodity=None,
                              clustering_method: str = 'weighted_kmeans',
                              max_clusters: int = None,
                              min_acres: int = None,
                              dataset: str = 'GHCND',
                              min_coverage: float = 0.9,
                              max_stations: int = 2,
                              export: bool = False,
                              create_visuals: bool = True,
                              run_id: str = None,
                              station_config_file=None) -> Dict:
        """
        Run the complete pipeline using agricultural density-based station location.
        
        This method identifies clusters of high agricultural activity and locates
        optimal weather stations for each cluster, providing better coverage of
        agriculturally important areas than grid-based approaches.
        
        Args:
            source: Data source (CSV file, 'nass', or NASS parameters)
            start_date: Start date for climate data
            end_date: End date for climate data
            commodity: Agricultural commodity for analysis (e.g., 'CORN', 'SOYBEANS')
            clustering_method: Method for agricultural clustering ('weighted_kmeans', 'kmeans', 'dbscan')
            max_clusters: Maximum number of agricultural clusters (auto-determined if None)
            min_acres: Minimum acres for cluster inclusion (auto-determined if None)
            dataset: NCEI dataset type ('GHCND' or 'GSOM')
            min_coverage: Minimum data coverage threshold for stations
            max_stations: Maximum stations per agricultural cluster
            export: Whether to export results to files
            create_visuals: Whether to create visualizations
            run_id: Unique identifier for this analysis run
            grid_size: [DEPRECATED] Use clustering_method instead
            density_threshold: [DEPRECATED] Use min_acres instead
            
        Returns:
            Dictionary with all pipeline results including agricultural analysis
        """
        import uuid

        if run_id is None:
            run_id = f"pipeline_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{str(uuid.uuid4())[:8]}"

        print("Starting Agricultural Climate Analysis Pipeline")
        print("=" * 60)
        print(f"Run ID: {run_id}")
        year = end_date.year if end_date.month > 5 else end_date.year - 1


        # Set intelligent defaults based on data size and coverage requirements


        # Store configuration for saving/loading
        self._clustering_method = clustering_method
        self._max_clusters = max_clusters
        self._min_acres = min_acres
        self._dataset = dataset
        self._start_date = start_date
        self._end_date = end_date
        self._min_coverage = min_coverage

        # Step 1: Load county data
        county_data = self.load_county_data(source, commodity=commodity, year=str(year))
        self.county_data = county_data  # Store for saving
        if max_clusters is None:
            max_clusters = max(6, min(15, len(county_data) // 50)) if hasattr(self, 'county_data') else 10
        if min_acres is None:
            min_acres = int(min_coverage * 50000)  # Scale with coverage requirement

        # Step 2: Locate stations by agricultural density
        print("\nStep 2: Locating weather stations by agricultural density...")
        agricultural_analysis = self.locate_stations_by_agricultural_density(
            clustering_method=clustering_method,
            max_clusters=max_clusters,
            min_acres=min_acres,
            max_stations_per_cluster=max_stations,
            dataset=dataset,
            start_date=start_date,
            end_date=end_date,
            min_coverage=min_coverage
        )
        self.agricultural_analysis = agricultural_analysis  # Store for saving
        
        # Store clusters and stations for configuration saving
        self.agricultural_clusters = agricultural_analysis.get('clusters', [])
        self.weather_stations = agricultural_analysis.get('stations', [])

        # Step 3: Fetch climate data from agricultural stations
        print("\nStep 3: Fetching climate data from agriculturally-optimized stations...")
        climate_data = self.fetch_climate_data_from_agricultural_stations(
            agricultural_analysis=agricultural_analysis,
            start_date=start_date,
            end_date=end_date,
            dataset=dataset
        )
        
        # Create agricultural extents for compatibility (mapping cluster centers to extents)
        self.agricultural_extents = self._create_agricultural_extents(agricultural_analysis)

        # Check if any climate data was retrieved
        if not climate_data:
            print("WARNING: No climate data was retrieved. Cannot calculate weather indices or create visualizations.")
            return {
                'run_id': run_id,
                'county_data': county_data,
                'agricultural_analysis': agricultural_analysis,
                'climate_data': {},
                'weather_indices': {},
                'agricultural_extents': self.agricultural_extents,
                'error': 'No climate data retrieved'
            }

        # Step 4: Calculate weather indices
        weather_indices = self.calculate_weather_indices(commodity=commodity)

        # Step 5: Create visualizations
        if create_visuals:
            print("\nCreating visualizations...")
            self.create_climate_visualizations(
                county_data=county_data,
                output_dir=f"visuals_{run_id}",
                show_plots=False
            )

        # Step 6: Export results
        if export:
            self.export_results(prefix=f"pipeline_{run_id}")

        print(f"\nSUCCESS: Agricultural Climate Analysis Pipeline completed! Run ID: {run_id}")
        print(f"Agricultural Summary:")
        print(f"  - Clusters analyzed: {len(agricultural_analysis.get('clusters', []))}")
        print(f"  - Stations used: {len(agricultural_analysis.get('stations', []))}")
        print(f"  - Total agricultural area: {agricultural_analysis.get('total_agricultural_area', 0):,.0f} acres")
        print(f"  - Climate datasets: {len(climate_data)}")
        print(f"  - Weather indices calculated: {len(weather_indices)}")
        if create_visuals:
            print(f"  - Visualizations: visuals_{run_id}/")

        return {
            'run_id': run_id,
            'county_data': county_data,
            'agricultural_analysis': agricultural_analysis,
            'climate_data': climate_data,
            'weather_indices': weather_indices,
            'agricultural_extents': self.agricultural_extents,
            'total_agricultural_area': agricultural_analysis.get('total_agricultural_area', 0),
            'clusters_analyzed': len(agricultural_analysis.get('clusters', [])),
            'stations_used': len(agricultural_analysis.get('stations', [])),
            'methodology': 'agricultural_density_clustering'
        }

    def save_station_configuration(self, save_path: str = None, include_climate_data: bool = True) -> str:
        """
        Save the complete weather station configuration and data.
        
        Args:
            save_path: Path to save configuration file (if None, auto-generated)
            include_climate_data: Whether to include raw climate data
            
        Returns:
            Path to the saved configuration file
        """
        import json
        
        if save_path is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            save_path = f"station_config_{timestamp}.json"
        
        print(f"Saving station configuration to: {save_path}")
        
        # Create configuration dictionary
        config = {
            'created': datetime.now().isoformat(),
            'pipeline_version': '2.0_agricultural_density',
            
            # Analysis parameters
            'clustering_method': getattr(self, '_clustering_method', 'weighted_kmeans'),
            'max_clusters': getattr(self, '_max_clusters', None),
            'min_acres': getattr(self, '_min_acres', None),
            'dataset': getattr(self, '_dataset', 'GHCND'),
            'min_coverage': getattr(self, '_min_coverage', 0.8),
            'date_range': {
                'start_date': getattr(self, '_start_date', None).isoformat() if hasattr(self, '_start_date') and self._start_date else None,
                'end_date': getattr(self, '_end_date', None).isoformat() if hasattr(self, '_end_date') and self._end_date else None
            },
            
            # Agricultural clusters (if available)
            'agricultural_clusters': [],
            'weather_stations': [],
            
            # Statistics
            'total_agricultural_area': 0,
            'station_count': 0,
            'cluster_count': 0
        }
        
        # Add agricultural clusters information
        if hasattr(self, 'agricultural_clusters') and self.agricultural_clusters:
            for cluster in self.agricultural_clusters:
                config['agricultural_clusters'].append({
                    'cluster_id': cluster.cluster_id,
                    'center_latitude': cluster.center_lat,
                    'center_longitude': cluster.center_lon,
                    'total_acres': cluster.total_acres,
                    'county_count': cluster.county_count,
                    'counties': cluster.counties,
                    'radius_km': cluster.radius_km,
                    'density_score': cluster.density_score
                })
            config['cluster_count'] = len(self.agricultural_clusters)
            config['total_agricultural_area'] = sum(c.total_acres for c in self.agricultural_clusters)
        
        # Add weather stations information
        if hasattr(self, 'weather_stations') and self.weather_stations:
            for station in self.weather_stations:
                config['weather_stations'].append({
                    'station_id': station.station_id,
                    'station_name': station.station_name,
                    'latitude': station.latitude,
                    'longitude': station.longitude,
                    'cluster_id': station.cluster_id,
                    'distance_km': station.distance_km,
                    'data_coverage': station.data_coverage,
                    'match_score': station.match_score
                })
            config['station_count'] = len(self.weather_stations)
        
        # Add climate data summary if available
        if include_climate_data and hasattr(self, 'climate_data') and self.climate_data:
            config['climate_data_summary'] = {}
            for cluster_name, df in self.climate_data.items():
                config['climate_data_summary'][cluster_name] = {
                    'record_count': len(df),
                    'date_range': [df['date'].min(), df['date'].max()] if 'date' in df.columns else None,
                    'variables': [col for col in df.columns if col not in ['station', 'date', 'latitude', 'longitude']],
                    'station_count': df['station'].nunique() if 'station' in df.columns else 0
                }
        
        # Save configuration to JSON file
        with open(save_path, 'w') as f:
            json.dump(config, f, indent=2, default=str)
        
        print(f"  Saved configuration: {len(config['agricultural_clusters'])} clusters, {len(config['weather_stations'])} stations")
        print(f"  Agricultural area: {config['total_agricultural_area']:,.0f} acres")
        print(f"SUCCESS: Station configuration saved to {save_path}")
        
        return save_path

    def load_station_configuration(self, config_path: str) -> Dict:
        """
        Load a previously saved weather station configuration.
        
        Args:
            config_path: Path to saved configuration file
            
        Returns:
            Dictionary with loaded configuration info
        """
        import json
        from .density_station_locator import DensityCluster as AgriculturalCluster, StationMatch
        
        config_file = Path(config_path)
        if not config_file.exists():
            raise FileNotFoundError(f"Station configuration file not found: {config_path}")
        
        print(f"Loading station configuration from: {config_path}")
        
        with open(config_file, 'r') as f:
            config = json.load(f)
        
        print(f"  Configuration created: {config.get('created', 'Unknown')}")
        print(f"  Pipeline version: {config.get('pipeline_version', 'Unknown')}")
        print(f"  Clusters: {len(config.get('agricultural_clusters', []))}")
        print(f"  Stations: {len(config.get('weather_stations', []))}")
        print(f"  Agricultural area: {config.get('total_agricultural_area', 0):,.0f} acres")
        
        # Restore configuration parameters
        self._clustering_method = config.get('clustering_method', 'weighted_kmeans')
        self._max_clusters = config.get('max_clusters')
        self._min_acres = config.get('min_acres')
        self._dataset = config.get('dataset', 'GHCND')
        self._min_coverage = config.get('min_coverage', 0.8)
        
        if config['date_range']['start_date']:
            self._start_date = datetime.fromisoformat(config['date_range']['start_date']).date()
        if config['date_range']['end_date']:
            self._end_date = datetime.fromisoformat(config['date_range']['end_date']).date()
        
        # Restore agricultural clusters
        if config.get('agricultural_clusters'):
            self.agricultural_clusters = []
            for cluster_data in config['agricultural_clusters']:
                cluster = AgriculturalCluster(
                    cluster_id=cluster_data['cluster_id'],
                    center_lat=cluster_data['center_latitude'],
                    center_lon=cluster_data['center_longitude'],
                    total_acres=cluster_data['total_acres'],
                    county_count=cluster_data['county_count'],
                    counties=cluster_data['counties'],
                    radius_km=cluster_data['radius_km'],
                    density_score=cluster_data['density_score']
                )
                self.agricultural_clusters.append(cluster)
            print(f"  Restored {len(self.agricultural_clusters)} agricultural clusters")
        
        # Restore weather stations
        if config.get('weather_stations'):
            self.weather_stations = []
            for station_data in config['weather_stations']:
                station = StationMatch(
                    station_id=station_data['station_id'],
                    station_name=station_data['station_name'],
                    latitude=station_data['latitude'],
                    longitude=station_data['longitude'],
                    cluster_id=station_data['cluster_id'],
                    distance_km=station_data['distance_km'],
                    data_coverage=station_data['data_coverage'],
                    match_score=station_data['match_score']
                )
                self.weather_stations.append(station)
            print(f"  Restored {len(self.weather_stations)} weather stations")
        
        print(f"SUCCESS: Station configuration loaded from {config_path}")
        
        return {
            'config': config,
            'clusters_loaded': len(config.get('agricultural_clusters', [])),
            'stations_loaded': len(config.get('weather_stations', [])),
            'total_agricultural_area': config.get('total_agricultural_area', 0),
            'load_path': str(config_file)
        }


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

def run_corn_pipeline(start_date: date, end_date: date,
                      csv_file: str = "counties_example.csv", 
                      min_coverage: float = 0.9,
                      clustering_method: str = 'weighted_kmeans',
                      max_clusters: int = None,
                      min_acres: int = None,
                      **kwargs) -> Dict:
    """
    Convenience function to run agricultural climate pipeline for corn data.
    
    Args:
        start_date: Start date for climate analysis
        end_date: End date for climate analysis
        csv_file: Path to county CSV file or 'nass' for API data
        min_coverage: Minimum station data coverage threshold
        clustering_method: Agricultural clustering method ('weighted_kmeans', 'kmeans', 'dbscan')
        max_clusters: Maximum agricultural clusters (auto-determined if None)
        min_acres: Minimum acres for cluster inclusion (auto-determined if None)
        **kwargs: Additional pipeline parameters
        
    Returns:
        Complete pipeline results with agricultural analysis
    """
    pipeline = ClimatePipeline()
    return pipeline.run_complete_pipeline(
        source=csv_file,
        start_date=start_date,
        end_date=end_date,
        commodity='CORN',
        clustering_method=clustering_method,
        max_clusters=max_clusters,
        min_acres=min_acres,
        min_coverage=min_coverage,
        **kwargs
    )


def run_nass_pipeline(start_date: date, end_date: date,
                      nass_params: Dict = None,
                      **kwargs) -> Dict:
    """
    Convenience function to run pipeline with NASS API data.
    
    Args:
        start_date: Start date for climate analysis
        end_date: End date for climate analysis
        nass_params: NASS API parameters
        **kwargs: Additional pipeline parameters
        
    Returns:
        Complete pipeline results
    """
    if nass_params is None:
        nass_params = {
            'short_desc': 'CORN - ACRES PLANTED',
            'agg_level_desc': 'COUNTY',
            'year': '2024'
        }

    pipeline = ClimatePipeline()
    return pipeline.run_complete_pipeline(
        source=nass_params,
        start_date=start_date,
        end_date=end_date,
        **kwargs
    )


# =============================================================================
# EXAMPLE USAGE
# =============================================================================


if __name__ == "__main__":
    from datetime import date

    print("Climate Grid Pipeline Example")
    print("=" * 40)

    # Example 1: Run with CSV data
    try:
        results = run_corn_pipeline(
            start_date=date(2024, 1, 1),
            end_date=date(2024, 12, 31),
            csv_file='nass',
            min_coverage=0.9,
            min_acres=35000,
            max_stations=3,
            max_clusters=6
        )

        print("\nPipeline Results Summary:")
        print(f"  Counties analyzed: {len(results['county_data'])}")
        print(f"  Grid cells created: {len(results['agricultural_analysis']['analysis_summary'])}")
        print(f"  Climate data sets: {len(results['climate_data'])}")
        print(f"  Weather indices: {len(results['weather_indices'])}")

    except Exception as e:
        print(f"ERROR: Error running pipeline: {str(e)}")

        # Fallback: Show pipeline structure
        print("\nPipeline Structure:")
        pipeline = ClimatePipeline()
        print("  1. Load county data (CSV or NASS API)")
        print("  2. Create climate grid from top acres grown")
        print("  3. Extract coordinate extents for each grid cell")
        print("  4. Fetch NCEI climate data using AgClimateAPI")
        print("  5. Calculate weather indices")
        print("  6. Export results")
