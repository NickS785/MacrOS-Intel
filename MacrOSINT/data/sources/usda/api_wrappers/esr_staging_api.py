"""
USDA Export Sales Report (ESR) Staging API Wrapper

This module provides a Python wrapper for accessing the USDA FAS staging API
for current ESR data. This API uses a different structure and parameters
compared to the production ESR API.

Author: Claude
Date: August 2025
Requirements: requests, pandas, aiohttp
"""

import requests
import pandas as pd
from datetime import datetime, timedelta
from typing import Optional, List, Dict, Any
import os
from pathlib import Path
from MacrOSINT import config
import toml
from io import StringIO


class USDAESRStagingError(Exception):
    """Custom exception for USDA ESR Staging API errors"""
    pass


class USDAESRStaging:
    """
    USDA Export Sales Report (ESR) Staging API Wrapper
    
    This class provides methods to access the staging ESR API for current data.
    The staging API has different endpoints and parameter structures compared
    to the production API.
    
    Example:
        esr = USDAESRStaging()
        
        # Get current week data for wheat to all countries
        current_data = esr.get_current_week_data(
            commodity_id=44,  # Wheat
            week_start_date="06/26/2025",
            week_end_date="06/26/2025"
        )
    """

    def __init__(self, base_url: str = "https://stgapps.fas.usda.gov", 
                 auth_token: Optional[str] = None):
        """
        Initialize the USDA ESR Staging API wrapper

        Args:
            base_url (str): Base URL for the staging API
            auth_token (str, optional): Bearer token for authorization
        """
        self.base_url = base_url.rstrip('/')
        self.api_key = os.getenv('FAS_TOKEN', config.FAS_TOKEN)
        
        # Use provided token or the token from your actual request
        self.auth_token = auth_token or "B2GWRPAFWB_qmq2oqSMUXsYZQaUIAVKSVplm-CbgKxItTcBtqDV0xHnEw40NOasf8kzt6AqWF4Gl6ltOLdgwRV0UdsTgRw2kcvQCtte8IZwEo9FYc5XtkF6arhc0-bRHod_9PPQYtmdZEgtRAiHK5FBPJD2_SJJG7Hel9Lte0IsoilIh-oJWqo3_wk-Gl4NmVwRiNPUdYIwFZCy0guEwrEBSYk30yvg_pwhnEDlUR-BAzo-5Ab20QyMtEHhilj3Ll9hfAJ3ucKikO724bFVAJWdUqOuGP-pW6rXRa2ocw0g2araaCGClVErkPbINx8sqwoYqHHn8-ry3ZhNDt8KMDGB4kKVUywRqjgyG2IS7S50YB_bIBnwkK4NasN2-rzyUk4NbrS4f6gn1tn7MnlojTg"
        
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.5',
            'Accept-Encoding': 'gzip, deflate, br, zstd',
            'Connection': 'keep-alive',
            'Upgrade-Insecure-Requests': '1',
            'Sec-Fetch-Dest': 'document',
            'Sec-Fetch-Mode': 'navigate',
            'Sec-Fetch-Site': 'none',
            'Sec-Fetch-User': '?1',
            'Priority': 'u=1',
            'Authorization': f'Bearer {self.auth_token}'
        })
        
        # Load commodity and country mappings from esr_map.toml
        self._load_mappings()

    def update_auth_token(self, new_token: str):
        """
        Update the authorization token for API requests
        
        Args:
            new_token (str): New bearer token for authorization
        """
        self.auth_token = new_token
        # Update session headers
        self.session.headers.update({
            'Authorization': f'Bearer {self.auth_token}'
        })
        print(f"Authorization token updated")

    def get_auth_status(self) -> Dict[str, str]:
        """
        Get authorization status information (token is masked for security)
        
        Returns:
            Dict with auth status information
        """
        if self.auth_token:
            masked_token = f"{self.auth_token[:8]}...{self.auth_token[-8:]}"
            return {
                "status": "configured",
                "token_preview": masked_token,
                "token_length": len(self.auth_token)
            }
        else:
            return {
                "status": "not_configured",
                "token_preview": "None",
                "token_length": 0
            }

    def _load_mappings(self):
        """Load commodity and country mappings from esr_map.toml"""
        try:
            esr_map_path = Path(__file__).parent.parent / "esr_map.toml"
            with open(esr_map_path) as f:
                esr_map = toml.load(f)
            
            self.esr_codes = esr_map.get('esr', {})
            self.commodities = self.esr_codes.get('commodities', {})
            self.countries = self.esr_codes.get('countries', {})
            self.aliases = self.esr_codes.get('alias', {})
            
            # Create reverse mappings for easier lookup
            self.commodity_name_to_code = self.commodities
            self.country_name_to_code = self.countries
            
            print(f"Loaded {len(self.commodities)} commodities and {len(self.countries)} countries from esr_map.toml")
            
        except Exception as e:
            print(f"Warning: Could not load esr_map.toml: {e}")
            self.esr_codes = {}
            self.commodities = {}
            self.countries = {}
            self.aliases = {}

    def _make_request(self, endpoint: str, params: Optional[Dict[str, Any]] = None,
                     method: str = "GET") -> pd.DataFrame:
        """
        Make a request to the staging ESR API and parse CSV response
        
        Args:
            endpoint (str): API endpoint
            params (dict, optional): Request parameters
            method (str): HTTP method (POST, GET)
            
        Returns:
            pd.DataFrame: Parsed CSV data as DataFrame
            
        Raises:
            USDAESRStagingError: If the API request fails
        """
        url = f"{self.base_url}/{endpoint.lstrip('/')}"
        
        try:
            print(f"Making {method} request to: {url}")  # Debug logging
            print(f"Request parameters: {params}")  # Debug logging
            
            if method.upper() == "POST":
                response = self.session.post(url, json=params)
            else:
                # For GET requests, convert arrays to comma-separated strings
                get_params = {}
                if params:
                    for key, value in params.items():
                        if isinstance(value, list):
                            # Convert list to comma-separated string for GET parameters
                            get_params[key] = ','.join(map(str, value))
                        else:
                            get_params[key] = value
                
                print(f"GET parameters: {get_params}")  # Debug logging
                response = self.session.get(url, params=get_params)
            
            print(f"Response status: {response.status_code}")  # Debug logging
            print(f"Response content type: {response.headers.get('content-type', 'unknown')}")
            
            response.raise_for_status()
            
            # Parse CSV response
            csv_content = response.text
            print(f"CSV response length: {len(csv_content)} characters")  # Debug logging
            
            if not csv_content.strip():
                print("Warning: Empty CSV response")
                return pd.DataFrame()
            
            # Parse CSV using pandas
            df = pd.read_csv(StringIO(csv_content))
            print(f"Parsed CSV: {df.shape[0]} rows, {df.shape[1]} columns")  # Debug logging
            
            if not df.empty:
                print(f"CSV columns: {list(df.columns)}")  # Debug logging
            
            return df
            
        except requests.exceptions.RequestException as e:
            raise USDAESRStagingError(f"API request failed: {str(e)} for URL: {url}")
        except pd.errors.EmptyDataError as e:
            print("Warning: Empty CSV data received")
            return pd.DataFrame()
        except Exception as e:
            raise USDAESRStagingError(f"Error parsing CSV response: {str(e)} for URL: {url}")

    def get_commodity_code(self, commodity_name: str) -> Optional[str]:
        """
        Get commodity code from commodity name using esr_map.toml
        
        Args:
            commodity_name: Name like 'cattle', 'wheat', 'soybeans'
            
        Returns:
            Commodity code string or None if not found
        """
        # First check if it's an alias (cattle -> beef)
        if commodity_name.lower() in self.aliases:
            actual_commodity = self.aliases[commodity_name.lower()]
        else:
            actual_commodity = commodity_name.lower()
        
        # Then get the commodity code
        return self.commodities.get(actual_commodity)

    def get_commodity_codes(self, commodity_names: List[str]) -> List[int]:
        """
        Get commodity codes from commodity names
        
        Args:
            commodity_names: List of commodity names
            
        Returns:
            List of commodity code integers
        """
        codes = []
        for name in commodity_names:
            code = self.get_commodity_code(name)
            if code:
                codes.append(int(code))
            else:
                print(f"Warning: Commodity '{name}' not found in esr_map.toml")
        
        return codes

    def get_country_codes(self, country_names: Optional[List[str]] = None) -> List[int]:
        """
        Get country codes from country names
        
        Args:
            country_names: List of country names, or None for all countries
            
        Returns:
            List of country code integers
        """
        if country_names is None:
            # Return all country codes as integers
            return [int(code) for code in self.countries.values()]
        
        codes = []
        for name in country_names:
            code = self.countries.get(name.upper())
            if code:
                codes.append(int(code))
            else:
                print(f"Warning: Country '{name}' not found in esr_map.toml")
        
        return codes

    def get_current_week_data(self,
                             commodity_name: str,
                             week_start_date: Optional[str] = None,
                             week_end_date: Optional[str] = None,
                             country_names: Optional[List[str]] = None) -> pd.DataFrame:
        """
        Get current week ESR data for a commodity
        
        Args:
            commodity_name: Commodity name (e.g., 'wheat', 'cattle', 'soybeans')
            week_start_date: Start date in MM/DD/YYYY format (defaults to current week)
            week_end_date: End date in MM/DD/YYYY format (defaults to current week) 
            country_names: List of country names (defaults to all countries)
            
        Returns:
            pd.DataFrame: ESR data for the specified parameters
        """
        # Get commodity code
        commodity_code = self.get_commodity_code(commodity_name)
        if not commodity_code:
            raise USDAESRStagingError(f"Unknown commodity: {commodity_name}")
        
        # Default to current week if dates not provided
        if not week_start_date or not week_end_date:
            today = datetime.now()
            # Find the Thursday of current week (ESR week ending date)
            days_until_thursday = (3 - today.weekday()) % 7
            thursday = today + timedelta(days=days_until_thursday)
            week_date = thursday.strftime("%m/%d/%Y")
            week_start_date = week_start_date or week_date
            week_end_date = week_end_date or week_date
        
        # Get country codes
        destination_codes = self.get_country_codes(country_names)
        
        # Prepare request parameters
        params = {
            "weekStartDate": week_start_date,
            "weekEndDate": week_end_date, 
            "commodityId": int(commodity_code),
            "destinationCode": destination_codes,
            "format": "Commodity/WeekEndingDate/ReportingMarketingYearName/Country"
        }
        
        print(f"Requesting data for commodity {commodity_name} (ID: {commodity_code})")
        print(f"Week: {week_start_date} to {week_end_date}")
        print(f"Countries: {len(destination_codes)} destinations")
        
        # Make the GET API request to the correct CSV endpoint
        df = self._make_request('/esrqs/api/reports/GetCsvFile', params=params, method="GET")
        
        # Convert date columns to datetime if they exist
        date_columns = ['weekEndingDate', 'reportDate', 'date', 'week_ending_date']
        for col in date_columns:
            if col in df.columns:
                df[col] = pd.to_datetime(df[col], errors='coerce')
        
        # Add metadata columns
        df['commodity_name'] = commodity_name
        df['api_source'] = 'staging'
        
        return df

    def get_multi_commodity_current_week_data(self,
                                             commodity_names: List[str],
                                             week_start_date: Optional[str] = None,
                                             week_end_date: Optional[str] = None,
                                             country_names: Optional[List[str]] = None) -> pd.DataFrame:
        """
        Get current week ESR data for multiple commodities in a single API request
        
        This is one of the key advantages of the staging API - it can handle multiple
        commodities and countries in a single request, which is much more efficient
        than making separate requests for each commodity.
        
        Args:
            commodity_names: List of commodity names (e.g., ['wheat', 'corn', 'soybeans'])
            week_start_date: Start date in MM/DD/YYYY format (defaults to current week)
            week_end_date: End date in MM/DD/YYYY format (defaults to current week)
            country_names: List of country names (defaults to all countries)
            
        Returns:
            pd.DataFrame: ESR data for all specified commodities and countries
        """
        # Get commodity codes
        commodity_codes = self.get_commodity_codes(commodity_names)
        if not commodity_codes:
            raise USDAESRStagingError(f"No valid commodities found in: {commodity_names}")
        
        # Default to current week if dates not provided
        if not week_start_date or not week_end_date:
            today = datetime.now()
            # Find the Thursday of current week (ESR week ending date)
            days_until_thursday = (3 - today.weekday()) % 7
            thursday = today + timedelta(days=days_until_thursday)
            week_date = thursday.strftime("%m/%d/%Y")
            week_start_date = week_start_date or week_date
            week_end_date = week_end_date or week_date
        
        # Get country codes
        destination_codes = self.get_country_codes(country_names)
        
        # Prepare request parameters for multiple commodities
        params = {
            "weekStartDate": week_start_date,
            "weekEndDate": week_end_date,
            "commodityId": commodity_codes,  # Multiple commodity IDs
            "destinationCode": destination_codes,
            "format": "Commodity/WeekEndingDate/ReportingMarketingYearName/Country"
        }
        
        print(f"Requesting data for multiple commodities: {commodity_names}")
        print(f"Commodity codes: {commodity_codes}")
        print(f"Week: {week_start_date} to {week_end_date}")
        print(f"Countries: {len(destination_codes)} destinations")
        
        # Make the GET API request to the correct CSV endpoint
        df = self._make_request('/esrqs/api/reports/GetCsvFile', params=params, method="GET")
        
        # Convert date columns to datetime if they exist
        date_columns = ['weekEndingDate', 'reportDate', 'date', 'week_ending_date']
        for col in date_columns:
            if col in df.columns:
                df[col] = pd.to_datetime(df[col], errors='coerce')
        
        # Add commodity name mapping if commodityId exists in the response
        if 'commodityId' in df.columns:
            # Create reverse mapping from commodity code to name
            code_to_name = {}
            for name in commodity_names:
                code = self.get_commodity_code(name)
                if code:
                    code_to_name[int(code)] = name
            
            # Map commodity codes back to names
            df['commodity_name'] = df['commodityId'].map(code_to_name)
        
        # Add metadata columns
        df['api_source'] = 'staging'
        df['request_type'] = 'multi_commodity'
        
        return df

    def list_available_commodities(self) -> List[str]:
        """List available commodities from esr_map.toml"""
        return list(self.commodities.keys())

    def list_available_countries(self) -> List[str]:
        """List available countries from esr_map.toml"""
        return list(self.countries.keys())


# Example usage and utility functions
if __name__ == "__main__":
    # Initialize the staging API wrapper
    esr = USDAESRStaging()
    
    # Example 1: Get current week data for wheat
    try:
        print("Getting current week data for wheat...")
        wheat_data = esr.get_current_week_data('wheat')
        print(f"Found {len(wheat_data)} wheat records")
        if not wheat_data.empty:
            print("\nColumns available:")
            print(wheat_data.columns.tolist())
            print("\nFirst few rows:")
            print(wheat_data.head())
    except USDAESRStagingError as e:
        print(f"Error getting wheat data: {e}")
    
    # Example 2: Get data for multiple commodities in single CSV request
    try:
        print("\nGetting current data for multiple commodities...")
        commodities = ['wheat', 'corn', 'soybeans']
        multi_data = esr.get_multi_commodity_current_week_data(commodities)
        
        print(f"Multi-commodity CSV data: {multi_data.shape}")
        if not multi_data.empty and 'commodity_name' in multi_data.columns:
            print("Commodities in data:")
            print(multi_data['commodity_name'].value_counts())
    except USDAESRStagingError as e:
        print(f"Error getting multi-commodity data: {e}")