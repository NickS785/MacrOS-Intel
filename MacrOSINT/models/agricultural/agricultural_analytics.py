"""
Commodity-Specific Time Series Analytics
=======================================

Specialized analytics functions for commodity data including:
- ESR (Export Sales Reporting) specific analysis
- Marketing year adjustments and seasonality
- Country/partner-based multi-series analysis
- Commitment vs actual export analysis
- Price-volume relationship modeling

Built on top of the general TimeSeriesAnalyzer with domain-specific enhancements.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Union, Any
from datetime import datetime, timedelta
import warnings

from ..timeseries_analysis import TimeSeriesAnalyzer

# Import plotting libraries if available
try:
    import plotly.graph_objects as go
    import plotly.express as px
    from plotly.subplots import make_subplots
    HAS_PLOTLY = True
except ImportError:
    HAS_PLOTLY = False
    warnings.warn("plotly not available - visualization functions will be disabled")


class ESRAnalyzer(TimeSeriesAnalyzer):
    """
    Specialized analyzer for Export Sales Reporting (ESR) data.
    
    Extends TimeSeriesAnalyzer with commodity-specific functionality:
    - Marketing year handling
    - Weekly export pattern analysis
    - Country/destination analysis
    - Commitment vs shipment analysis
    """
    
    # Standard ESR columns
    ESR_COLUMNS = {
        'weekly_exports': 'weeklyExports',
        'outstanding_sales': 'outstandingSales', 
        'gross_new_sales': 'grossNewSales',
        'current_my_net_sales': 'currentMYNetSales',
        'current_my_total_commitment': 'currentMYTotalCommitment',
        'next_my_outstanding_sales': 'nextMYOutstandingSales',
        'next_my_net_sales': 'nextMYNetSales',
        'date': 'weekEndingDate',
        'country': 'country',
        'commodity': 'commodity'
    }
    
    @staticmethod
    def aggregate_multi_country_data(data: pd.DataFrame, countries: list, 
                                    group_cols: list = None) -> pd.DataFrame:
        """
        Static method to aggregate data from multiple countries by summing values.
        
        Args:
            data: ESR DataFrame with country columns_col
            countries: List of countries to include in aggregation
            group_cols: Columns to group by (default: ['weekEndingDate', 'marketing_year', 'my_week'])
            
        Returns:
            Aggregated DataFrame with summed values for selected countries
        """
        if group_cols is None:
            group_cols = ['weekEndingDate']
            # Add marketing_year and my_week if they exist
            if 'marketing_year' in data.columns:
                group_cols.append('marketing_year')
            if 'my_week' in data.columns:
                group_cols.append('my_week')
        
        # Filter data for selected countries
        country_data = data[data['country'].isin(countries)].copy()
        
        if country_data.empty:
            return pd.DataFrame()
        
        # Define numeric columns to sum (exclude grouping and categorical columns)
        exclude_cols = group_cols + ['country', 'commodity']
        numeric_cols = [col for col in country_data.select_dtypes(include=[np.number]).columns 
                       if col not in exclude_cols]
        
        # Group by specified columns and sum numeric values
        agg_dict = {col: 'sum' for col in numeric_cols}
        
        # Add non-numeric columns we want to preserve
        if 'commodity' in country_data.columns:
            agg_dict['commodity'] = 'first'
        
        aggregated = country_data.groupby(group_cols).agg(agg_dict).reset_index()
        
        # Add a combined country identifier
        aggregated['country'] = f'Combined ({len(countries)} countries)'
        
        return aggregated

    def __init__(self, data: pd.DataFrame, commodity_type: str = 'grains'):
        """
        Initialize ESR analyzer.
        
        Args:
            data: DataFrame with ESR data
            commodity_type: 'grains', 'oilseeds', or 'livestock' for marketing year logic
        """
        self.commodity_type = commodity_type
        self.marketing_year_start = self._get_marketing_year_start(commodity_type)
        
        # Standardize columns_col names if needed
        data_copy = self._standardize_esr_columns(data)
        
        # Initialize parent class with only numeric value columns
        numeric_esr_columns = [
            self.ESR_COLUMNS['weekly_exports'],
            self.ESR_COLUMNS['outstanding_sales'],
            self.ESR_COLUMNS['gross_new_sales'],
            self.ESR_COLUMNS['current_my_net_sales'],
            self.ESR_COLUMNS['current_my_total_commitment'],
            self.ESR_COLUMNS['next_my_outstanding_sales'],
            self.ESR_COLUMNS['next_my_net_sales']
        ]
        
        # Filter to only include columns that actually exist in the data
        available_numeric_columns = [col for col in numeric_esr_columns if col in data_copy.columns]
        
        super().__init__(
            data_copy, 
            date_column=self.ESR_COLUMNS['date'],
            value_columns=available_numeric_columns
        )
        
        # Add derived columns
        self._add_derived_columns()
    
    def _standardize_esr_columns(self, data: pd.DataFrame) -> pd.DataFrame:
        """Standardize ESR columns_col names to expected format."""
        data_copy = data.copy()
        
        # Map common variations to standard names
        column_mappings = {
            'week_ending_date': self.ESR_COLUMNS['date'],
            'weekendingdate': self.ESR_COLUMNS['date'],
            'weekly_export': self.ESR_COLUMNS['weekly_exports'],
            'weeklyexport': self.ESR_COLUMNS['weekly_exports'],
            'outstanding_sale': self.ESR_COLUMNS['outstanding_sales'],
            'outstandingsale': self.ESR_COLUMNS['outstanding_sales']
        }
        
        # Apply mappings (case insensitive)
        for old_name, new_name in column_mappings.items():
            for col in data_copy.columns:
                if col.lower() == old_name.lower():
                    data_copy = data_copy.rename(columns={col: new_name})
                    break
        
        return data_copy
    
    def _get_marketing_year_start(self, commodity_type: str) -> int:
        """Get marketing year start month for commodity type."""
        marketing_years = {
            'grains': 9,      # September (wheat, corn, etc.)
            'oilseeds': 9,    # September (soybeans, etc.)
            'livestock': 1    # January (cattle, pork, etc.)
        }
        return marketing_years.get(commodity_type, 9)
    
    def _add_derived_columns(self):
        """Add derived columns for analysis."""
        # Marketing year
        self.data['marketing_year'] = self.data.apply(
            lambda row: self._get_marketing_year(row.name), axis=1
        )
        
        # Week of marketing year
        self.data['my_week'] = self.data.apply(
            lambda row: self._get_my_week(row.name), axis=1
        )
        
        # Export efficiency (exports / outstanding sales)
        if 'outstandingSales' in self.data.columns and 'weeklyExports' in self.data.columns:
            self.data['export_efficiency'] = (
                self.data['weeklyExports'] / 
                self.data['outstandingSales'].replace(0, np.nan)
            ).fillna(0)
        
        # Sales momentum (change in gross new sales)
        if 'grossNewSales' in self.data.columns:
            self.data['sales_momentum'] = self.data['grossNewSales'].diff()
    
    def _get_marketing_year(self, date: datetime) -> int:
        """Calculate marketing year for a given date."""
        if date.month >= self.marketing_year_start:
            return date.year
        else:
            return date.year - 1
    
    def _get_my_week(self, date: datetime) -> int:
        """Calculate week of marketing year."""
        my_start = datetime(self._get_marketing_year(date), self.marketing_year_start, 1)
        delta = date - my_start
        return max(1, delta.days // 7 + 1)
    
    def analyze_seasonal_patterns(self, metric: str = 'weeklyExports',
                                 group_by: Optional[str] = None) -> Dict[str, Any]:
        """
        Analyze seasonal patterns in ESR data.
        
        Args:
            metric: ESR metric to analyze
            group_by: Group by 'country', 'commodity', or None
            
        Returns:
            Dictionary with seasonal analysis results
        """
        if group_by:
            return self._group_seasonal_analysis(metric, group_by)
        
        data = self.data[[metric, 'my_week', 'marketing_year']].dropna()
        
        # Weekly patterns within marketing year
        weekly_patterns = data.groupby('my_week')[metric].agg(['mean', 'std', 'count'])
        peak_weeks = weekly_patterns['mean'].nlargest(5).index.tolist()
        low_weeks = weekly_patterns['mean'].nsmallest(5).index.tolist()
        
        # Year-over-year comparison
        yearly_totals = data.groupby('marketing_year')[metric].sum()
        yearly_growth = yearly_totals.pct_change().dropna()
        
        # Seasonality strength (coefficient of variation of weekly means)
        seasonality_strength = weekly_patterns['mean'].std() / weekly_patterns['mean'].mean()
        
        return {
            'weekly_patterns': weekly_patterns.to_dict(),
            'peak_weeks': peak_weeks,
            'low_weeks': low_weeks,
            'yearly_totals': yearly_totals.to_dict(),
            'average_growth_rate': yearly_growth.mean(),
            'seasonality_strength': seasonality_strength,
            'total_weeks_analyzed': len(weekly_patterns)
        }
    
    def _group_seasonal_analysis(self, metric: str, group_by: str) -> Dict[str, Any]:
        """Seasonal analysis grouped by country/commodity."""
        results = {}
        
        for group_name, group_data in self.data.groupby(group_by):
            if len(group_data) >= 52:  # At least one marketing year
                group_analyzer = ESRAnalyzer(group_data.reset_index(), self.commodity_type)
                results[str(group_name)] = group_analyzer.analyze_seasonal_patterns(metric)
            else:
                results[str(group_name)] = {'error': 'insufficient_data'}
        
        return results
    
    def commitment_vs_shipment_analysis(self, country: Optional[str] = None,
                                      countries: Optional[list] = None,
                                      commodity: Optional[str] = None) -> Dict[str, Any]:
        """
        Analyze relationship between commitments and actual shipments.
        
        Args:
            country: Specific single country to analyze (optional)
            countries: List of countries to aggregate and analyze (optional)
            commodity: Specific commodity to analyze (optional)
            
        Returns:
            Dictionary with commitment analysis results
        """
        data = self.data.copy()
        
        # Handle multi-country aggregation
        if countries and len(countries) > 1:
            data = self.aggregate_multi_country_data(data, countries)
        elif countries and len(countries) == 1:
            data = data[data['country'] == countries[0]]
        elif country:
            data = data[data['country'] == country]
            
        if commodity:
            data = data[data['commodity'] == commodity]
        
        if len(data) == 0:
            return {'error': 'no_data_after_filtering'}
        
        # Key metrics for analysis
        required_cols = ['weeklyExports', 'outstandingSales', 'currentMYTotalCommitment']
        available_cols = [col for col in required_cols if col in data.columns]
        
        if len(available_cols) < 2:
            return {'error': 'insufficient_columns'}
        
        results = {}
        
        # Export fulfillment rate (weekly exports / outstanding sales)
        if all(col in data.columns for col in ['weeklyExports', 'outstandingSales']):
            data['fulfillment_rate'] = (
                data['weeklyExports'] / data['outstandingSales'].replace(0, np.nan)
            ).fillna(0)
            
            results['fulfillment_rate'] = {
                'mean': data['fulfillment_rate'].mean(),
                'median': data['fulfillment_rate'].median(),
                'std': data['fulfillment_rate'].std(),
                'trend': self._calculate_trend(data['fulfillment_rate'].dropna())
            }
        
        # Commitment utilization (exports / total commitment)
        if all(col in data.columns for col in ['weeklyExports', 'currentMYTotalCommitment']):
            data['commitment_utilization'] = (
                data['weeklyExports'] / data['currentMYTotalCommitment'].replace(0, np.nan)
            ).fillna(0)
            
            results['commitment_utilization'] = {
                'mean': data['commitment_utilization'].mean(),
                'median': data['commitment_utilization'].median(),
                'std': data['commitment_utilization'].std(),
                'trend': self._calculate_trend(data['commitment_utilization'].dropna())
            }
        
        # Sales backlog (outstanding sales / recent average exports)
        if all(col in data.columns for col in ['weeklyExports', 'outstandingSales']):
            recent_avg_exports = data['weeklyExports'].rolling(window=4).mean()
            data['sales_backlog_weeks'] = (
                data['outstandingSales'] / recent_avg_exports.replace(0, np.nan)
            ).fillna(0)
            
            # Also create a simplified sales_backlog columns_col for charting
            data['sales_backlog'] = data['sales_backlog_weeks']
            
            results['sales_backlog'] = {
                'mean_weeks': data['sales_backlog_weeks'].mean(),
                'median_weeks': data['sales_backlog_weeks'].median(),
                'max_weeks': data['sales_backlog_weeks'].max(),
                'trend': self._calculate_trend(data['sales_backlog_weeks'].dropna())
            }
        results['data'] = data
        
        # Correlations between commitment metrics
        correlation_cols = [col for col in available_cols if col in data.columns]
        if len(correlation_cols) > 1:
            corr_matrix = data[correlation_cols].corr()
            results['correlations'] = corr_matrix.to_dict()
        
        return results
    
    def country_performance_ranking(self, metric: str = 'weeklyExports',
                                  time_period: str = 'recent_year') -> pd.DataFrame:
        """
        Rank countries by export performance metrics.
        
        Args:
            metric: ESR metric to rank by
            time_period: 'recent_year', 'all_time', or specific year
            
        Returns:
            DataFrame with country rankings
        """
        data = self.data.copy()
        
        # Filter by time period
        if time_period == 'recent_year':
            recent_year = data['marketing_year'].max()
            data = data[data['marketing_year'] == recent_year]
        elif isinstance(time_period, int):
            data = data[data['marketing_year'] == time_period]
        
        if 'country' not in data.columns:
            raise ValueError("Country columns_col not found in data")
        
        # Aggregate by country
        country_stats = data.groupby('country')[metric].agg([
            'sum', 'mean', 'count', 'std'
        ]).round(2)
        
        # Calculate additional metrics
        country_stats['total'] = country_stats['sum']
        country_stats['average'] = country_stats['mean']
        country_stats['weeks_active'] = country_stats['count']
        country_stats['volatility'] = country_stats['std'] / country_stats['mean']
        country_stats['volatility'] = country_stats['volatility'].fillna(0)
        
        # Calculate market share
        total_market = country_stats['total'].sum()
        country_stats['market_share_pct'] = (country_stats['total'] / total_market * 100).round(2)
        
        # Rank by total
        country_stats['rank_by_total'] = country_stats['total'].rank(method='dense', ascending=False)
        country_stats['rank_by_average'] = country_stats['average'].rank(method='dense', ascending=False)
        
        # Sort by total descending
        result = country_stats.sort_values('total', ascending=False)
        
        # Clean up columns
        result = result[['total', 'average', 'market_share_pct', 'weeks_active', 
                        'volatility', 'rank_by_total', 'rank_by_average']]
        
        return result
    
    def detect_export_anomalies(self, metric: str = 'weeklyExports',
                               method: str = 'seasonal_adjusted',
                               group_by: Optional[str] = None) -> pd.DataFrame:
        """
        Detect anomalies in export patterns.
        
        Args:
            metric: ESR metric to analyze for anomalies
            method: 'seasonal_adjusted', 'iqr', or 'zscore'
            group_by: Group analysis by 'country' or 'commodity'
            
        Returns:
            DataFrame with anomaly flags and scores
        """
        if method == 'seasonal_adjusted':
            return self._detect_seasonal_anomalies(metric, group_by)
        else:
            return self._detect_statistical_anomalies(metric, method, group_by)
    
    def _detect_seasonal_anomalies(self, metric: str, group_by: Optional[str]) -> pd.DataFrame:
        """Detect anomalies using seasonal adjustment."""
        results = []
        
        if group_by:
            for group_name, group_data in self.data.groupby(group_by):
                if len(group_data) >= 52:  # Need at least 1 year
                    group_analyzer = ESRAnalyzer(group_data.reset_index(), self.commodity_type)
                    
                    # Seasonal normalize the metric
                    normalized = group_analyzer.seasonal_normalize(metric, method='stl')
                    
                    # Detect outliers in normalized data
                    outliers = group_analyzer.detect_outliers(metric, method='zscore')
                    
                    group_results = pd.DataFrame({
                        'date': group_data.index,
                        'original_value': group_data[metric],
                        'seasonal_adjusted': normalized,
                        'is_anomaly': outliers[group_data.index],
                        'anomaly_score': np.abs(normalized - normalized.mean()) / normalized.std(),
                        group_by: group_name
                    })
                    results.append(group_results)
        else:
            # Single series analysis
            normalized = self.seasonal_normalize(metric, method='stl')
            outliers = self.detect_outliers(metric, method='zscore')
            
            results.append(pd.DataFrame({
                'date': self.data.index,
                'original_value': self.data[metric],
                'seasonal_adjusted': normalized,
                'is_anomaly': outliers,
                'anomaly_score': np.abs(normalized - normalized.mean()) / normalized.std()
            }))
        
        return pd.concat(results, ignore_index=True) if results else pd.DataFrame()
    
    def _detect_statistical_anomalies(self, metric: str, method: str, 
                                    group_by: Optional[str]) -> pd.DataFrame:
        """Detect anomalies using statistical methods."""
        if group_by:
            outliers = self.detect_outliers(metric, method=method, group_by=group_by)
        else:
            outliers = self.detect_outliers(metric, method=method)
        
        data_subset = self.data[[metric]].copy()
        if group_by and group_by in self.data.columns:
            data_subset[group_by] = self.data[group_by]
        
        data_subset['is_anomaly'] = outliers
        data_subset['date'] = self.data.index
        
        return data_subset.reset_index(drop=True)
    
    def export_forecast_simple(self, metric: str = 'weeklyExports',
                             periods: int = 8, 
                             country: Optional[str] = None) -> pd.DataFrame:
        """
        Simple export forecasting using seasonal patterns.
        
        Args:
            metric: ESR metric to forecast
            periods: Number of weeks to forecast
            country: Specific country to forecast (optional)
            
        Returns:
            DataFrame with forecast values
        """
        data = self.data.copy()
        
        if country:
            data = data[data['country'] == country]
        
        if len(data) < 52:
            return pd.DataFrame({'error': ['insufficient_data']})
        
        # Get seasonal patterns
        seasonal_analysis = self.analyze_seasonal_patterns(metric)
        weekly_means = pd.Series(seasonal_analysis['weekly_patterns']['mean'])
        
        # Get recent trend
        recent_data = data[metric].tail(12)  # Last 3 months
        trend = self._calculate_trend(recent_data)
        
        # Generate forecast dates
        last_date = data.index.max()
        forecast_dates = pd.date_range(
            start=last_date + timedelta(weeks=1),
            periods=periods,
            freq='W'
        )
        
        # Generate forecasts
        forecasts = []
        for i, date in enumerate(forecast_dates):
            my_week = self._get_my_week(date)
            
            # Get seasonal component
            seasonal_value = weekly_means.get(my_week, weekly_means.mean())
            
            # Apply trend
            trend_component = trend * (i + 1)
            
            # Combine components
            forecast = seasonal_value + trend_component
            
            forecasts.append({
                'date': date,
                'forecast': max(0, forecast),  # Ensure non-negative
                'seasonal_component': seasonal_value,
                'trend_component': trend_component,
                'my_week': my_week
            })
        
        return pd.DataFrame(forecasts)
    
    def generate_seasonal_overlay(self, metric: str, start_year: int, end_year: int) -> pd.DataFrame:
        """
        Generate seasonal overlay data for multiple marketing years.
        
        Args:
            metric: ESR metric to overlay (e.g., 'weeklyExports')
            start_year: Starting marketing year
            end_year: Ending marketing year
            
        Returns:
            DataFrame with columns: my_week, marketing_year, and metric values
        """
        # Filter data for the specified marketing year range
        data = self.data[
            (self.data['marketing_year'] >= start_year) & 
            (self.data['marketing_year'] <= end_year)
        ].copy()
        
        if data.empty:
            return pd.DataFrame()
        
        # Select relevant columns for overlay
        overlay_columns = ['my_week', 'marketing_year', metric]
        available_columns = [col for col in overlay_columns if col in data.columns]
        
        if metric not in data.columns:
            return pd.DataFrame()
        
        # Create the overlay DataFrame
        overlay_data = data[available_columns].copy()
        
        # Remove rows with null values in the metric
        overlay_data = overlay_data.dropna(subset=[metric])
        
        # Convert marketing_year to string for better plotting
        overlay_data['marketing_year'] = overlay_data['marketing_year'].astype(str)
        
        # Sort by marketing year and week for consistent plotting
        overlay_data = overlay_data.sort_values(['marketing_year', 'my_week'])
        
        return overlay_data
    
    def generate_seasonal_analysis_data(self, metric: str, marketing_year: Optional[int] = None) -> Dict[str, Any]:
        """
        Generate seasonal analysis data formatted for dashboard callbacks.
        
        Args:
            metric: ESR metric to analyze (e.g., 'weeklyExports')
            marketing_year: Optional specific marketing year to analyze
            
        Returns:
            Dictionary with 'data' key containing DataFrame and analysis results
        """
        # Filter data for specific marketing year if provided
        if marketing_year:
            filtered_data = self.data[self.data['marketing_year'] == marketing_year].copy()
        else:
            filtered_data = self.data.copy()
            
        if filtered_data.empty:
            return {
                'error': 'no_data_available',
                'data': pd.DataFrame()
            }
        
        # Ensure we have the required columns
        if metric not in filtered_data.columns:
            return {
                'error': f'metric_{metric}_not_found',
                'data': pd.DataFrame()
            }
        
        # Generate seasonal patterns analysis
        seasonal_patterns = self.analyze_seasonal_patterns(metric)
        
        # Add additional derived columns for plotting
        filtered_data = filtered_data.copy()
        
        # Calculate rolling averages for trend analysis
        if len(filtered_data) >= 4:
            filtered_data['rolling_avg_4w'] = filtered_data[metric].rolling(window=4, min_periods=1).mean()
        if len(filtered_data) >= 8:
            filtered_data['rolling_avg_8w'] = filtered_data[metric].rolling(window=8, min_periods=1).mean()
        
        # Calculate seasonal normalized values if we have enough data
        if 'weekly_patterns' in seasonal_patterns and len(filtered_data) > 12:
            weekly_means = pd.Series(seasonal_patterns['weekly_patterns']['mean'])
            
            # Create seasonal adjustment based on weekly patterns
            filtered_data['seasonal_component'] = filtered_data['my_week'].map(weekly_means)
            filtered_data['seasonal_adjusted'] = filtered_data[metric] - filtered_data['seasonal_component'].fillna(0)
            
            # Calculate detrended values
            if len(filtered_data) >= 12:
                trend = filtered_data[metric].rolling(window=12, center=True).mean()
                filtered_data['detrended'] = filtered_data[metric] - trend
        
        return {
            'data': filtered_data,
            'seasonal_patterns': seasonal_patterns,
            'marketing_year': marketing_year,
            'metric': metric,
            'data_points': len(filtered_data)
        }


class CommodityPriceAnalyzer(TimeSeriesAnalyzer):
    """
    Specialized analyzer for commodity price data with economic indicators.
    """
    
    def __init__(self, data: pd.DataFrame, price_columns: List[str]):
        """
        Initialize commodity price analyzer.
        
        Args:
            data: DataFrame with price and economic data
            price_columns: List of price columns to analyze
        """
        super().__init__(data, value_columns=price_columns)
        self.price_columns = price_columns
    
    def volatility_analysis(self, column: str, window: int = 30) -> pd.DataFrame:
        """
        Calculate rolling volatility measures.
        
        Args:
            column: Price columns_col to analyze
            window: Rolling window for volatility calculation
            
        Returns:
            DataFrame with volatility measures
        """
        data = self.data[column].dropna()
        returns = data.pct_change().dropna()
        
        result = pd.DataFrame(index=data.index)
        result['price'] = data
        result['returns'] = returns.reindex(result.index)
        result['volatility'] = returns.rolling(window=window).std() * np.sqrt(252)  # Annualized
        result['realized_vol'] = returns.rolling(window=window).apply(
            lambda x: np.sqrt(np.sum(x**2) * 252)
        )
        
        return result
    
    def price_momentum_indicators(self, column: str) -> pd.DataFrame:
        """
        Calculate price momentum indicators.
        
        Args:
            column: Price columns_col to analyze
            
        Returns:
            DataFrame with momentum indicators
        """
        data = self.data[column].dropna()
        
        result = pd.DataFrame(index=data.index)
        result['price'] = data
        
        # Moving averages
        result['ma_10'] = data.rolling(10).mean()
        result['ma_30'] = data.rolling(30).mean()
        result['ma_50'] = data.rolling(50).mean()
        
        # Momentum oscillators
        result['roc_10'] = data.pct_change(10) * 100  # 10-period rate of change
        result['roc_30'] = data.pct_change(30) * 100
        
        # Price relative to moving averages
        result['price_vs_ma10'] = (data / result['ma_10'] - 1) * 100
        result['price_vs_ma30'] = (data / result['ma_30'] - 1) * 100
        
        return result


def create_esr_analyzer(data: pd.DataFrame, commodity_type: str = 'grains') -> ESRAnalyzer:
    """
    Factory function to create ESR analyzer.
    
    Args:
        data: ESR DataFrame
        commodity_type: 'grains', 'oilseeds', or 'livestock'
        
    Returns:
        Configured ESRAnalyzer instance
    """
    return ESRAnalyzer(data, commodity_type)


# Utility functions for common commodity analysis tasks
def compare_country_exports(esr_data: pd.DataFrame, countries: List[str],
                          metric: str = 'weeklyExports') -> Dict[str, Any]:
    """Compare export performance across countries."""
    analyzer = ESRAnalyzer(esr_data)
    
    results = {}
    for country in countries:
        country_data = esr_data[esr_data['country'] == country]
        if len(country_data) > 0:
            country_analyzer = ESRAnalyzer(country_data.reset_index())
            results[country] = {
                'total_exports': country_data[metric].sum(),
                'average_weekly': country_data[metric].mean(),
                'peak_week': country_data[metric].max(),
                'volatility': country_data[metric].std() / country_data[metric].mean(),
                'seasonal_patterns': country_analyzer.analyze_seasonal_patterns(metric)
            }
    
    return results


def marketing_year_comparison(esr_data: pd.DataFrame, 
                            years: List[int],
                            metric: str = 'weeklyExports') -> pd.DataFrame:
    """Compare metrics across marketing years."""
    analyzer = ESRAnalyzer(esr_data)
    
    results = []
    for year in years:
        year_data = esr_data[esr_data['marketing_year'] == year]
        if len(year_data) > 0:
            results.append({
                'marketing_year': year,
                'total': year_data[metric].sum(),
                'average_weekly': year_data[metric].mean(),
                'peak_weekly': year_data[metric].max(),
                'weeks_with_data': len(year_data),
                'volatility': year_data[metric].std()
            })
    
    comparison = pd.DataFrame(results)
    
    # Calculate year-over-year changes
    if len(comparison) > 1:
        comparison['yoy_total_change'] = comparison['total'].pct_change()
        comparison['yoy_avg_change'] = comparison['average_weekly'].pct_change()
    
    return comparison.set_index('marketing_year')


if __name__ == "__main__":
    # Example usage with sample ESR data
    print("Commodity analytics module loaded successfully!")
    print("Available classes: ESRAnalyzer, CommodityPriceAnalyzer")
    print("Use create_esr_analyzer() for quick ESR analysis setup")