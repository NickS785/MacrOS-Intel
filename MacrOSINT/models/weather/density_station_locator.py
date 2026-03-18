"""
General Density-Based Station Locator Module
============================================

This module provides a general-purpose class for locating weather stations based on 
density data from any DataFrame with FIPS codes, county names, and density values.
Can be used for agricultural density, population density, economic activity, etc.

Author: Agricultural Climate Analysis Team
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Tuple
from datetime import date
from dataclasses import dataclass
import warnings

# Clustering and spatial analysis
from sklearn.cluster import DBSCAN, KMeans
from sklearn.preprocessing import StandardScaler

# Optional imports for enhanced geospatial analysis
try:
    import geopandas as gpd
    from shapely.geometry import Point
    HAS_GEOPANDAS = True
except ImportError:
    HAS_GEOPANDAS = False
    warnings.warn("geopandas not available. Some spatial features will be limited.")

from .agclimate import AgClimateAPI


@dataclass
class DensityCluster:
    """Represents a cluster of high density activity."""
    cluster_id: int
    center_lat: float
    center_lon: float
    total_density: float
    county_count: int
    counties: List[str]
    radius_km: float
    density_score: float
    fips_codes: List[str]


@dataclass
class StationMatch:
    """Represents a weather station matched to a density cluster."""
    station_id: str
    station_name: str
    latitude: float
    longitude: float
    cluster_id: int
    distance_km: float
    data_coverage: float
    match_score: float


class DensityBasedLocator:
    """
    General-purpose class for locating optimal weather stations based on density data.
    Works with any DataFrame containing FIPS codes, coordinates, and density values.
    """
    
    def __init__(self, ncei_token: str = None):
        """
        Initialize the station locator.
        
        Args:
            ncei_token: NCEI API token for station data retrieval
        """
        from MacrOSINT.config import NCEI_TOKEN
        self.ncei_token = ncei_token or NCEI_TOKEN
        self.agclimate_api = AgClimateAPI(self.ncei_token) if self.ncei_token else None
        self.clusters = []
        self.stations = []
        self.matched_stations = []
        self.density_column = None
        self.county_column = None
        self.lat_column = None
        self.lon_column = None
        self.fips_column = None
    
    def identify_density_clusters(self, 
                                data: pd.DataFrame,
                                density_column: str,
                                county_column: str = 'county_name',
                                lat_column: str = 'latitude',
                                lon_column: str = 'longitude',
                                fips_column: str = 'fips',
                                method: str = 'weighted_kmeans',
                                min_density_threshold: float = None,
                                max_clusters: int = 15,
                                **kwargs) -> List[DensityCluster]:
        """
        Identify clusters of high density activity from any density data.
        
        Args:
            data: DataFrame with density data and geographic information
            density_column: Name of column containing density values
            county_column: Name of column containing county names
            lat_column: Name of column containing latitude values
            lon_column: Name of column containing longitude values
            fips_column: Name of column containing FIPS codes
            method: Clustering method ('dbscan', 'kmeans', 'weighted_kmeans')
            min_density_threshold: Minimum density threshold for inclusion
            max_clusters: Maximum number of clusters to create
            **kwargs: Additional parameters for clustering algorithms
            
        Returns:
            List of DensityCluster objects
        """
        print(f"Identifying density clusters using {method} method...")
        print(f"Density column: {density_column}")
        
        # Store column names for later use
        self.density_column = density_column
        self.county_column = county_column
        self.lat_column = lat_column
        self.lon_column = lon_column
        self.fips_column = fips_column
        
        # Validate required columns
        required_cols = [density_column, county_column, lat_column, lon_column]
        missing_cols = [col for col in required_cols if col not in data.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")
        
        # Set default density threshold if not provided
        if min_density_threshold is None:
            min_density_threshold = data[density_column].quantile(0.5)  # Median as default
        
        # Filter for counties with significant density
        high_density = data[data[density_column] >= min_density_threshold].copy()
        
        if len(high_density) < 3:
            print(f"WARNING: Only {len(high_density)} counties meet minimum density threshold")
            return []
        
        print(f"  Analyzing {len(high_density)} high-density counties")
        print(f"  Total density in analysis: {high_density[density_column].sum():,.0f}")
        print(f"  Density threshold: {min_density_threshold:,.0f}")
        
        # Prepare data for clustering
        if method == 'dbscan':
            clusters = self._dbscan_clustering(high_density, **kwargs)
        elif method == 'kmeans':
            clusters = self._kmeans_clustering(high_density, max_clusters, **kwargs)
        elif method == 'weighted_kmeans':
            clusters = self._weighted_kmeans_clustering(high_density, max_clusters, **kwargs)
        else:
            raise ValueError(f"Unknown clustering method: {method}")
        
        self.clusters = clusters
        
        # Print cluster summary
        print(f"\nCluster Analysis Results:")
        print(f"  Total clusters identified: {len(clusters)}")
        for i, cluster in enumerate(clusters):
            print(f"  Cluster {i+1}: {cluster.county_count} counties, "
                  f"{cluster.total_density:,.0f} density units, "
                  f"center at ({cluster.center_lat:.2f}, {cluster.center_lon:.2f})")
        
        return clusters
    
    def _dbscan_clustering(self, data: pd.DataFrame, **kwargs) -> List[DensityCluster]:
        """Apply DBSCAN clustering based on geographic proximity and density."""
        
        # Prepare features: lat, lon, and log-scaled density
        features = np.column_stack([
            data[self.lat_column],
            data[self.lon_column],
            np.log1p(data[self.density_column])  # Log scale for density
        ])
        
        # Scale features
        scaler = StandardScaler()
        features_scaled = scaler.fit_transform(features)
        
        # Apply DBSCAN
        eps = kwargs.get('eps', 0.5)  # Distance threshold
        min_samples = kwargs.get('min_samples', 3)  # Minimum counties per cluster
        
        dbscan = DBSCAN(eps=eps, min_samples=min_samples)
        cluster_labels = dbscan.fit_predict(features_scaled)
        
        return self._create_cluster_objects(data, cluster_labels)
    
    def _kmeans_clustering(self, data: pd.DataFrame, n_clusters: int, **kwargs) -> List[DensityCluster]:
        """Apply K-means clustering."""
        
        # Prepare features
        features = np.column_stack([
            data[self.lat_column],
            data[self.lon_column]
        ])
        
        # Determine optimal number of clusters if not specified
        if n_clusters is None:
            n_clusters = min(8, len(data) // 5)  # Heuristic: ~5 counties per cluster
        
        n_clusters = min(n_clusters, len(data))
        
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, **kwargs)
        cluster_labels = kmeans.fit_predict(features)
        
        return self._create_cluster_objects(data, cluster_labels)
    
    def _weighted_kmeans_clustering(self, data: pd.DataFrame, n_clusters: int, **kwargs) -> List[DensityCluster]:
        """Apply weighted K-means clustering where counties with higher density have more influence."""
        
        # Prepare features with density weighting
        features = np.column_stack([
            data[self.lat_column],
            data[self.lon_column]
        ])
        
        # Use density as sample weights
        weights = data[self.density_column].values
        weights = weights / weights.sum()  # Normalize
        
        n_clusters = min(n_clusters or 8, len(data))
        
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, **kwargs)
        
        # Weighted clustering: repeat samples based on their weight
        weighted_features = []
        weighted_indices = []
        
        for i, (row, weight) in enumerate(zip(features, weights)):
            # Number of repetitions based on weight (minimum 1)
            reps = max(1, int(weight * len(data) * 2))
            weighted_features.extend([row] * reps)
            weighted_indices.extend([i] * reps)
        
        weighted_features = np.array(weighted_features)
        cluster_labels_weighted = kmeans.fit_predict(weighted_features)
        
        # Map back to original data
        cluster_labels = np.zeros(len(data), dtype=int)
        for i, orig_idx in enumerate(weighted_indices):
            cluster_labels[orig_idx] = cluster_labels_weighted[i]
        
        return self._create_cluster_objects(data, cluster_labels)
    
    def _create_cluster_objects(self, data: pd.DataFrame, cluster_labels: np.ndarray) -> List[DensityCluster]:
        """Create DensityCluster objects from clustering results."""
        
        clusters = []
        unique_labels = np.unique(cluster_labels)
        
        for label in unique_labels:
            if label == -1:  # DBSCAN noise points
                continue
            
            cluster_mask = cluster_labels == label
            cluster_data = data[cluster_mask]
            
            if len(cluster_data) == 0:
                continue
            
            # Calculate weighted center based on density values
            weights = cluster_data[self.density_column].values
            center_lat = np.average(cluster_data[self.lat_column], weights=weights)
            center_lon = np.average(cluster_data[self.lon_column], weights=weights)
            
            # Calculate cluster radius (95th percentile distance from center)
            distances = np.sqrt(
                (cluster_data[self.lat_column] - center_lat) ** 2 +
                (cluster_data[self.lon_column] - center_lon) ** 2
            ) * 111  # Convert degrees to km (approximate)
            
            radius_km = np.percentile(distances, 95) if len(distances) > 1 else 5.0
            
            # Calculate density score (density per square km)
            area_km2 = np.pi * radius_km ** 2
            density_score = cluster_data[self.density_column].sum() / area_km2
            
            # Get FIPS codes if available
            fips_codes = []
            if self.fips_column and self.fips_column in cluster_data.columns:
                fips_codes = cluster_data[self.fips_column].tolist()
            
            cluster = DensityCluster(
                cluster_id=len(clusters),
                center_lat=center_lat,
                center_lon=center_lon,
                total_density=cluster_data[self.density_column].sum(),
                county_count=len(cluster_data),
                counties=cluster_data[self.county_column].tolist(),
                radius_km=radius_km,
                density_score=density_score,
                fips_codes=fips_codes
            )
            
            clusters.append(cluster)
        
        # Sort clusters by total density (descending)
        clusters.sort(key=lambda x: x.total_density, reverse=True)
        
        # Update cluster IDs to reflect ranking
        for i, cluster in enumerate(clusters):
            cluster.cluster_id = i
        
        return clusters
    
    def find_optimal_stations(self,
                            clusters: List[DensityCluster],
                            dataset: str = 'GHCND',
                            variables: List[str] = None,
                            start_date: date = None,
                            end_date: date = None,
                            min_coverage: float = 0.8,
                            max_stations_per_cluster: int = 2,
                            search_radius_km: float = 50) -> List[StationMatch]:
        """
        Find optimal weather stations for each density cluster.
        
        Args:
            clusters: List of density clusters
            dataset: NCEI dataset to search ('GHCND' or 'GSOM')
            variables: Climate variables needed
            start_date: Start date for data coverage check
            end_date: End date for data coverage check
            min_coverage: Minimum data coverage threshold
            max_stations_per_cluster: Maximum stations per cluster
            search_radius_km: Search radius around cluster center (km)
            
        Returns:
            List of StationMatch objects
        """
        if not self.agclimate_api:
            raise ValueError("NCEI API token required for station search")
        
        print(f"Finding optimal stations for {len(clusters)} clusters...")
        
        variables = variables or ['TMAX', 'TMIN', 'PRCP']
        matched_stations = []
        
        for cluster in clusters:
            print(f"\nSearching for stations near Cluster {cluster.cluster_id} "
                  f"(density: {cluster.total_density:,.0f})...")
            
            # Create search extent (bounding box around cluster)
            search_radius_deg = search_radius_km / 111  # Convert km to degrees
            extent = f"{cluster.center_lat - search_radius_deg}," \
                    f"{cluster.center_lon - search_radius_deg}," \
                    f"{cluster.center_lat + search_radius_deg}," \
                    f"{cluster.center_lon + search_radius_deg}"
            
            try:
                # Get stations in the area
                stations_response = self.agclimate_api.ncei.get_stations(
                    datasetid=dataset,
                    datatypeid=variables,
                    extent=extent,
                    startdate=start_date,
                    enddate=end_date,
                    limit=1000
                )
                
                if not stations_response:
                    print(f"  No stations found for cluster {cluster.cluster_id}")
                    continue
                
                stations_df = stations_response.to_dataframe()
                
                # Filter by coverage threshold
                quality_stations = stations_df[
                    stations_df['datacoverage'] >= min_coverage
                ].copy()
                
                if quality_stations.empty:
                    print(f"  No stations meet coverage threshold for cluster {cluster.cluster_id}")
                    continue
                
                # Calculate distances and match scores
                distances = np.sqrt(
                    (quality_stations['latitude'] - cluster.center_lat) ** 2 +
                    (quality_stations['longitude'] - cluster.center_lon) ** 2
                ) * 111  # Convert to km
                
                quality_stations['distance_km'] = distances
                
                # Calculate match score (weighted combination of coverage and proximity)
                coverage_score = quality_stations['datacoverage']
                distance_score = 1 - (distances / search_radius_km)  # Closer = better
                match_score = 0.7 * coverage_score + 0.3 * distance_score
                
                quality_stations['match_score'] = match_score
                
                # Select top stations for this cluster
                top_stations = quality_stations.nlargest(max_stations_per_cluster, 'match_score')
                
                for _, station in top_stations.iterrows():
                    station_match = StationMatch(
                        station_id=station['id'],
                        station_name=station['name'],
                        latitude=station['latitude'],
                        longitude=station['longitude'],
                        cluster_id=cluster.cluster_id,
                        distance_km=station['distance_km'],
                        data_coverage=station['datacoverage'],
                        match_score=station['match_score']
                    )
                    matched_stations.append(station_match)
                
                print(f"  Found {len(top_stations)} stations for cluster {cluster.cluster_id}")
                
            except Exception as e:
                print(f"  Error searching stations for cluster {cluster.cluster_id}: {e}")
                continue
        
        # Sort by match score (best first)
        matched_stations.sort(key=lambda x: x.match_score, reverse=True)
        
        self.matched_stations = matched_stations
        
        print(f"\nStation Matching Complete:")
        print(f"  Total stations found: {len(matched_stations)}")
        if matched_stations:
            print(f"  Average coverage: {np.mean([s.data_coverage for s in matched_stations]):.1%}")
            print(f"  Average distance: {np.mean([s.distance_km for s in matched_stations]):.1f} km")
        
        return matched_stations
    
    def export_results(self, 
                      clusters: List[DensityCluster],
                      stations: List[StationMatch],
                      output_prefix: str = "density_stations") -> Dict[str, str]:
        """
        Export clusters and stations to CSV files.
        
        Args:
            clusters: Density clusters
            stations: Matched weather stations
            output_prefix: Prefix for output files
            
        Returns:
            Dictionary mapping file types to file paths
        """
        files_created = {}
        
        # Export clusters
        if clusters:
            clusters_data = []
            for cluster in clusters:
                clusters_data.append({
                    'cluster_id': cluster.cluster_id,
                    'center_latitude': cluster.center_lat,
                    'center_longitude': cluster.center_lon,
                    'total_density': cluster.total_density,
                    'county_count': cluster.county_count,
                    'radius_km': cluster.radius_km,
                    'density_score': cluster.density_score,
                    'counties': '; '.join(cluster.counties),
                    'fips_codes': '; '.join(cluster.fips_codes) if cluster.fips_codes else ''
                })
            
            clusters_file = f"{output_prefix}_clusters.csv"
            pd.DataFrame(clusters_data).to_csv(clusters_file, index=False)
            files_created['clusters'] = clusters_file
            print(f"Clusters exported to: {clusters_file}")
        
        # Export stations
        if stations:
            stations_data = []
            for station in stations:
                stations_data.append({
                    'station_id': station.station_id,
                    'station_name': station.station_name,
                    'latitude': station.latitude,
                    'longitude': station.longitude,
                    'cluster_id': station.cluster_id,
                    'distance_km': station.distance_km,
                    'data_coverage': station.data_coverage,
                    'match_score': station.match_score
                })
            
            stations_file = f"{output_prefix}_stations.csv"
            pd.DataFrame(stations_data).to_csv(stations_file, index=False)
            files_created['stations'] = stations_file
            print(f"Stations exported to: {stations_file}")
        
        return files_created


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

def find_stations_for_agricultural_areas(county_data: pd.DataFrame,
                                        clustering_method: str = 'weighted_kmeans',
                                        max_clusters: int = 12,
                                        min_acres: int = 50000,
                                        dataset: str = 'GHCND',
                                        start_date: date = None,
                                        end_date: date = None,
                                        min_coverage: float = 0.8,
                                        max_stations_per_cluster: int = 2,
                                        ncei_token: str = None) -> Tuple[List[DensityCluster], List[StationMatch]]:
    """
    Convenience function for agricultural data using the general density locator.
    Maintains backward compatibility with existing agricultural workflows.
    
    Args:
        county_data: DataFrame with agricultural county data
        clustering_method: Method for identifying clusters
        max_clusters: Maximum number of clusters
        min_acres: Minimum acres for cluster inclusion
        dataset: NCEI dataset type
        start_date: Start date for coverage analysis
        end_date: End date for coverage analysis
        min_coverage: Minimum data coverage threshold
        max_stations_per_cluster: Maximum stations per cluster
        ncei_token: NCEI API token
        
    Returns:
        Tuple of (clusters, matched_stations)
    """
    return find_stations_for_density_data(
        data=county_data,
        density_column='acres_planted',
        county_column='county_name',
        lat_column='latitude',
        lon_column='longitude',
        fips_column='fips',
        clustering_method=clustering_method,
        max_clusters=max_clusters,
        min_density_threshold=min_acres,
        dataset=dataset,
        start_date=start_date,
        end_date=end_date,
        min_coverage=min_coverage,
        max_stations_per_cluster=max_stations_per_cluster,
        ncei_token=ncei_token
    )

def find_stations_for_density_data(data: pd.DataFrame,
                                  density_column: str,
                                  county_column: str = 'county_name',
                                  lat_column: str = 'latitude',
                                  lon_column: str = 'longitude',
                                  fips_column: str = 'fips',
                                  clustering_method: str = 'weighted_kmeans',
                                  max_clusters: int = 12,
                                  min_density_threshold: float = None,
                                  dataset: str = 'GHCND',
                                  start_date: date = None,
                                  end_date: date = None,
                                  min_coverage: float = 0.8,
                                  max_stations_per_cluster: int = 2,
                                  ncei_token: str = None) -> Tuple[List[DensityCluster], List[StationMatch]]:
    """
    Complete workflow to find optimal weather stations for density-based areas.
    
    Args:
        data: DataFrame with density data and geographic information
        density_column: Name of column containing density values
        county_column: Name of column containing county names
        lat_column: Name of column containing latitude values
        lon_column: Name of column containing longitude values
        fips_column: Name of column containing FIPS codes
        clustering_method: Method for identifying clusters
        max_clusters: Maximum number of clusters
        min_density_threshold: Minimum density for cluster inclusion
        dataset: NCEI dataset type
        start_date: Start date for coverage analysis
        end_date: End date for coverage analysis
        min_coverage: Minimum data coverage threshold
        max_stations_per_cluster: Maximum stations per cluster
        ncei_token: NCEI API token
        
    Returns:
        Tuple of (clusters, matched_stations)
    """
    # Initialize locator
    locator = DensityBasedLocator(ncei_token)
    
    # Identify clusters
    clusters = locator.identify_density_clusters(
        data=data,
        density_column=density_column,
        county_column=county_column,
        lat_column=lat_column,
        lon_column=lon_column,
        fips_column=fips_column,
        method=clustering_method,
        min_density_threshold=min_density_threshold,
        max_clusters=max_clusters
    )
    
    if not clusters:
        print("No density clusters identified")
        return [], []
    
    # Find optimal stations
    stations = locator.find_optimal_stations(
        clusters=clusters,
        dataset=dataset,
        start_date=start_date,
        end_date=end_date,
        min_coverage=min_coverage,
        max_stations_per_cluster=max_stations_per_cluster
    )
    
    # Export results
    locator.export_results(clusters, stations, f'density_{density_column}')
    
    return clusters, stations