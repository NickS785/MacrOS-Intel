"""
Models Package
==============

Time series analysis and commodity-specific analytics for the commodities dashboard.

Available modules:
- timeseries_analysis: General time series analysis toolkit
- commodity_analytics: Specialized ESR and commodity analysis
- seasonal: Basic seasonal index utilities

Quick start examples:

# ESR Analysis
from models.commodity_analytics import create_esr_analyzer
analyzer = create_esr_analyzer(esr_data, commodity_type='grains')
seasonal_patterns = analyzer.analyze_seasonal_patterns('weeklyExports', group_by='country')

# General Time Series Analysis  
from models.timeseries_analysis import TimeSeriesAnalyzer
analyzer = TimeSeriesAnalyzer(data, date_column='date', value_columns=['price', 'volume'])
normalized = analyzer.seasonal_normalize('price', method='stl')
stationarity = analyzer.test_stationarity('price')
"""

from .timeseries_analysis import TimeSeriesAnalyzer
from MacrOSINT.models.agricultural.agricultural_analytics import ESRAnalyzer, CommodityPriceAnalyzer, create_esr_analyzer
from .seasonal import create_seasonal_index, seasonal_difference, get_seasonal_ratio

# Utility functions
from MacrOSINT.models.agricultural.agricultural_analytics import compare_country_exports, marketing_year_comparison

__all__ = [
    'TimeSeriesAnalyzer',
    'ESRAnalyzer', 
    'CommodityPriceAnalyzer',
    'create_esr_analyzer',
    'compare_country_exports',
    'marketing_year_comparison',
    'create_seasonal_index',
    'seasonal_difference',
    'get_seasonal_ratio'
]

__version__ = '1.0.0'
