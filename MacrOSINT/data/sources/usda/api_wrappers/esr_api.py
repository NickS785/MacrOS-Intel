"""
USDA Export Sales Report (ESR) API Wrapper

This module provides a Python wrapper for accessing the USDA Foreign Agricultural Service's 
Export Sales Report API. It allows easy retrieval of agricultural commodity export data,
including export sales, commodities, countries, and related metadata.

Author: Claude
Date: July 2025
Requirements: requests, pandas
"""

import requests
import pandas as pd
from datetime import datetime, date
from typing import Optional, Union, List, Dict, Any
from dotenv import load_dotenv
import json
import os
from MacrOSINT import config
import asyncio
import aiohttp


class USDAESRError(Exception):
    """Custom exception for USDA ESR API errors"""
    pass





if not os.getenv('FAS_TOKEN', False):
    load_dotenv(config.DOT_ENV)


class USDAESR:
    """
    USDA Export Sales Report (ESR) API Wrapper

    This class provides methods to access various ESR API endpoints including:
    - Export sales data
    - Commodities information
    - Countries and regions
    - Units of measure
    - Data release dates

    Example:
        esr = USDAESR()

        # Get all commodities
        commodities = esr.get_commodities()

        # Get export sales data for wheat to specific countries
        wheat_exports = esr.get_export_sales(
            commodity_code=1,  # Wheat
            country_code=[5040, 5530],  # Canada and Mexico
            start_date='2023-01-01',
            end_date='2023-12-31'
        )
    """

    def __init__(self, base_url: str = "https://api.fas.usda.gov/api/esr"):
        """
        Initialize the USDA ESR API wrapper

        Args:
            base_url (str): Base URL for the ESR API
        """
        self.base_url = base_url.rstrip('/')
        self.api_key = os.getenv('FAS_TOKEN', config.FAS_TOKEN)  # Store API key as instance variable
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'USDA-ESR-Python-Wrapper/1.0',
            'Accept': 'application/json',
            'Content-Type': 'application/json',
            'X-Api-Key': self.api_key
        })

    def _make_request(self, endpoint: str, params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Make a request to the ESR API

        Args:
            endpoint (str): API endpoint
            params (dict, optional): Query parameters

        Returns:
            dict: JSON response from the API

        Raises:
            USDAESRError: If the API request fails
        """
        url = f"{self.base_url}/{endpoint.lstrip('/')}"
        req_path = str()
        if params:
            if params.get('commodityCode'):
                req_path += f'/commodityCode/{params.get("commodityCode")}'
            if params.get('countryCode'):
                req_path += f'/countryCode/{params.get("countryCode")}'
            elif params.get('regionCode'):
                req_path += f'/regionCode/{params.get("regionCode")}'
            if params.get('marketYear'):
                req_path += f'/marketYear/{params.get("marketYear")}'

        # Clean up URL construction
        if url.endswith('/'):
            url = url[:-1]
        req_url = url + req_path

        try:
            print(f"Making sync request to: {req_url}")  # Debug logging
            response = self.session.get(req_url)
            print(f"Response status: {response.status_code}")  # Debug logging
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            raise USDAESRError(f"API request failed: {str(e)} for URL: {req_url}")
        except json.JSONDecodeError as e:
            raise USDAESRError(f"Invalid JSON response: {str(e)} for URL: {req_url}")

    def get_commodities(self) -> pd.DataFrame:
        """
        Get all available commodities

        Returns:
            pd.DataFrame: DataFrame containing commodity information
        """
        data = self._make_request('/commodities')
        return pd.DataFrame(data)

    def get_countries(self) -> pd.DataFrame:
        """
        Get all available countries

        Returns:
            pd.DataFrame: DataFrame containing country information
        """
        data = self._make_request('/countries')
        return pd.DataFrame(data)

    def get_regions(self) -> pd.DataFrame:
        """
        Get all available regions

        Returns:
            pd.DataFrame: DataFrame containing region information
        """
        data = self._make_request('/regions')
        return pd.DataFrame(data)

    def get_units_of_measure(self) -> pd.DataFrame:
        """
        Get all available units of measure

        Returns:
            pd.DataFrame: DataFrame containing units of measure
        """
        data = self._make_request('/unitsOfMeasure')
        return pd.DataFrame(data)

    def get_data_release_dates(self) -> pd.DataFrame:
        """
        Get data release dates

        Returns:
            pd.DataFrame: DataFrame containing data release dates
        """
        data = self._make_request('/datareleasedates')
        return pd.DataFrame(data)

    def get_export_sales(self,
                         commodity_code: Optional[Union[int, List[int]]] = None,
                         country_code: Optional[Union[int, List[int]]] = None,
                         region_code: Optional[Union[int, List[int]]] = None,
                         market_year: Optional[Union[int, List[int]]] = None,
                         start_date: Optional[Union[str, date, datetime]] = None,
                         end_date: Optional[Union[str, date, datetime]] = None,
                         unit_id: Optional[int] = None) -> pd.DataFrame:
        """
        Get export sales data with optional filtering

        Args:
            commodity_code (int or list, optional): Commodity code(s) to filter by
            country_code (int or list, optional): Country code(s) to filter by
            region_code (int or list, optional): Region code(s) to filter by
            market_year (int or list, optional): Marketing year(s) to filter by
            start_date (str/date/datetime, optional): Start date for data (YYYY-MM-DD format)
            end_date (str/date/datetime, optional): End date for data (YYYY-MM-DD format)
            unit_id (int, optional): Unit of measure ID

        Returns:
            pd.DataFrame: DataFrame containing export sales data

        Example:
            # Get wheat exports to Canada and Mexico for 2023
            wheat_data = esr.get_export_sales(
                commodity_code=1,
                country_code=[5040, 5530],
                start_date='2023-01-01',
                end_date='2023-12-31'
            )
        """
        params = {}

        # Handle commodity codes
        if commodity_code is not None:
            if isinstance(commodity_code, list):
                params['commodityCode'] = ','.join(map(str, commodity_code))
            else:
                params['commodityCode'] = str(commodity_code)

        # Handle country codes
        if country_code is not None:
            if isinstance(country_code, list):
                params['countryCode'] = ','.join(map(str, country_code))
            else:
                params['countryCode'] = str(country_code)

        # Handle region codes
        if region_code is not None:
            if isinstance(region_code, list):
                params['regionCode'] = ','.join(map(str, region_code))
            else:
                params['regionCode'] = str(region_code)

        # Handle marketing year
        if market_year is not None:
            if isinstance(market_year, list):
                params['marketYear'] = ','.join(map(str, market_year))
            else:
                params['marketYear'] = str(market_year)

        # Handle start date
        if start_date is not None:
            if isinstance(start_date, (date, datetime)):
                params['startDate'] = start_date.strftime('%Y-%m-%d')
            else:
                params['startDate'] = str(start_date)

        # Handle end date
        if end_date is not None:
            if isinstance(end_date, (date, datetime)):
                params['endDate'] = end_date.strftime('%Y-%m-%d')
            else:
                params['endDate'] = str(end_date)

        # Handle unit ID
        if unit_id is not None:
            params['unitId'] = str(unit_id)

        # Make the request - try common endpoint names
        try:
            data = self._make_request('/exports', params=params)
        except USDAESRError:
            try:
                data = self._make_request('/data', params=params)
            except USDAESRError:
                try:
                    data = self._make_request('/sales', params=params)
                except USDAESRError:
                    # If all else fails, try the base endpoint with parameters
                    data = self._make_request('/', params=params)

        df = pd.DataFrame(data)

        # Convert date columns to datetime if they exist
        date_columns = ['weekEndingDate', 'reportDate', 'date', 'week_ending_date']
        for col in date_columns:
            if col in df.columns:
                df[col] = pd.to_datetime(df[col], errors='coerce')

        return df

    def search_commodities(self, search_term: str) -> pd.DataFrame:
        """
        Search for commodities by name

        Args:
            search_term (str): Term to search for in commodity names

        Returns:
            pd.DataFrame: Filtered DataFrame containing matching commodities
        """
        commodities = self.get_commodities()
        if 'commodityName' in commodities.columns:
            mask = commodities['commodityName'].str.contains(search_term, case=False, na=False)
        elif 'name' in commodities.columns:
            mask = commodities['name'].str.contains(search_term, case=False, na=False)
        else:
            # If columns_col names are different, search all string columns
            mask = commodities.select_dtypes(include=['object']).apply(
                lambda x: x.str.contains(search_term, case=False, na=False)
            ).any(axis=1)

        return commodities[mask]

    def search_countries(self, search_term: str) -> pd.DataFrame:
        """
        Search for countries by name

        Args:
            search_term (str): Term to search for in country names

        Returns:
            pd.DataFrame: Filtered DataFrame containing matching countries
        """
        countries = self.get_countries()
        if 'countryName' in countries.columns:
            mask = countries['countryName'].str.contains(search_term, case=False, na=False)
        elif 'name' in countries.columns:
            mask = countries['name'].str.contains(search_term, case=False, na=False)
        else:
            # If columns_col names are different, search all string columns
            mask = countries.select_dtypes(include=['object']).apply(
                lambda x: x.str.contains(search_term, case=False, na=False)
            ).any(axis=1)

        return countries[mask]

    def get_commodity_exports_by_country(self,
                                         commodity_code: int,
                                         market_year: Optional[int] = None) -> pd.DataFrame:
        """
        Get export sales data for a specific commodity, grouped by country

        Args:
            commodity_code (int): Commodity code
            market_year (int, optional): Marketing year to filter by

        Returns:
            pd.DataFrame: Export sales data grouped by country
        """
        data = self.get_export_sales(
            commodity_code=commodity_code,
            market_year=market_year
        )

        if not data.empty and 'countryCode' in data.columns:
            # Group by country and sum numeric columns
            numeric_cols = data.select_dtypes(include=['int64', 'float64']).columns
            grouped = data.groupby('countryCode')[numeric_cols].sum()

            # Add country names if available
            try:
                countries = self.get_countries()
                if 'countryCode' in countries.columns and 'countryName' in countries.columns:
                    grouped = grouped.merge(
                        countries[['countryCode', 'countryName']],
                        on='countryCode',
                        how='left'
                    )
            except:
                pass  # Continue without country names if merge fails

            return grouped

        return data

    async def _make_request_async(self, session: aiohttp.ClientSession, endpoint: str,
                                  params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Make an async request to the ESR API
        
        Args:
            session: aiohttp ClientSession
            endpoint: API endpoint
            params: Query parameters
            
        Returns:
            dict: JSON response from the API
            
        Raises:
            USDAESRError: If the API request fails
        """
        url = f"{self.base_url}/{endpoint.lstrip('/')}"
        req_path = str()

        if params:
            if params.get('commodityCode'):
                req_path += f'/commodityCode/{params.get("commodityCode")}'
            if params.get('countryCode'):
                req_path += f'/countryCode/{params.get("countryCode")}'
            elif params.get('regionCode'):
                req_path += f'/regionCode/{params.get("regionCode")}'
            if params.get('marketYear'):
                req_path += f'/marketYear/{params.get("marketYear")}'

        # Clean up URL construction
        if url.endswith('/'):
            url = url[:-1]
        req_url = url + req_path

        # Prepare headers for aiohttp (convert from requests session headers)
        headers = {
            'User-Agent': 'USDA-ESR-Python-Wrapper/1.0',
            'Accept': 'application/json',
            'Content-Type': 'application/json',
            'X-Api-Key': self.api_key
        }

        try:
            print(f"Making async request to: {req_url}")  # Debug logging
            async with session.get(req_url, headers=headers) as response:
                print(f"Response status: {response.status}")  # Debug logging
                response.raise_for_status()
                return await response.json()
        except aiohttp.ClientError as e:
            raise USDAESRError(f"API request failed: {str(e)} for URL: {req_url}")
        except json.JSONDecodeError as e:
            raise USDAESRError(f"Invalid JSON response: {str(e)} for URL: {req_url}")

    async def get_export_sales_async(self,
                                     session: aiohttp.ClientSession,
                                     commodity_code: Optional[Union[int, List[int]]] = None,
                                     country_code: Optional[Union[int, List[int]]] = None,
                                     region_code: Optional[Union[int, List[int]]] = None,
                                     market_year: Optional[Union[int, List[int]]] = None,
                                     start_date: Optional[Union[str, date, datetime]] = None,
                                     end_date: Optional[Union[str, date, datetime]] = None,
                                     unit_id: Optional[int] = None) -> pd.DataFrame:
        """
        Async version of get_export_sales for concurrent requests
        
        Args:
            session: aiohttp ClientSession for the request
            commodity_code, country_code, etc.: Same as get_export_sales
            
        Returns:
            pd.DataFrame: DataFrame containing export sales data
        """
        params = {}

        # Handle commodity codes
        if commodity_code is not None:
            if isinstance(commodity_code, list):
                params['commodityCode'] = ','.join(map(str, commodity_code))
            else:
                params['commodityCode'] = str(commodity_code)

        # Handle country codes
        if country_code is not None:
            if isinstance(country_code, list):
                params['countryCode'] = ','.join(map(str, country_code))
            else:
                params['countryCode'] = str(country_code)

        # Handle region codes
        if region_code is not None:
            if isinstance(region_code, list):
                params['regionCode'] = ','.join(map(str, region_code))
            else:
                params['regionCode'] = str(region_code)

        # Handle marketing year
        if market_year is not None:
            if isinstance(market_year, list):
                params['marketYear'] = ','.join(map(str, market_year))
            else:
                params['marketYear'] = str(market_year)

        # Handle start date
        if start_date is not None:
            if isinstance(start_date, (date, datetime)):
                params['startDate'] = start_date.strftime('%Y-%m-%d')
            else:
                params['startDate'] = str(start_date)

        # Handle end date
        if end_date is not None:
            if isinstance(end_date, (date, datetime)):
                params['endDate'] = end_date.strftime('%Y-%m-%d')
            else:
                params['endDate'] = str(end_date)

        # Handle unit ID
        if unit_id is not None:
            params['unitId'] = str(unit_id)

        # Make the async request - try common endpoint names
        data = None
        for endpoint in ['/exports', '/data', '/sales', '/']:
            try:
                data = await self._make_request_async(session, endpoint, params=params)
                break
            except USDAESRError:
                continue

        if data is None:
            raise USDAESRError("Failed to retrieve data from any known endpoint")

        df = pd.DataFrame(data)

        # Convert date columns to datetime if they exist
        date_columns = ['weekEndingDate', 'reportDate', 'date', 'week_ending_date']
        for col in date_columns:
            if col in df.columns:
                df[col] = pd.to_datetime(df[col], errors='coerce')

        return df

    async def get_multi_year_exports_async(self,
                                           commodity_code: Union[int, List[int]],
                                           years: List[int],
                                           country_codes: Optional[List[int]] = None,
                                           max_concurrent: int = 3) -> Dict[int, pd.DataFrame]:
        """
        Get export sales data for multiple years concurrently
        
        Args:
            commodity_code: Commodity code(s)
            years: List of marketing years to retrieve
            country_codes: Optional list of country codes to filter
            max_concurrent: Maximum concurrent requests
            
        Returns:
            Dict mapping year to DataFrame of export data
        """
        results = {}
        semaphore = asyncio.Semaphore(max_concurrent)

        async def fetch_year_data(session: aiohttp.ClientSession, year: int) -> tuple[int, pd.DataFrame]:
            """Fetch data for a single year with concurrency control"""
            async with semaphore:
                try:
                    print(f"Fetching ESR data for year {year}...")
                    data = await self.get_export_sales_async(
                        session=session,
                        commodity_code=commodity_code,
                        country_code=country_codes,
                        market_year=year
                    )
                    print(f"✓ Retrieved {len(data)} records for year {year}")
                    return year, data
                except Exception as e:
                    print(f"✗ Failed to retrieve data for year {year}: {e}")
                    return year, pd.DataFrame()  # Return empty DataFrame on error

        # Create aiohttp session with proper timeout and connector settings
        timeout = aiohttp.ClientTimeout(total=30)
        connector = aiohttp.TCPConnector(limit=max_concurrent)

        async with aiohttp.ClientSession(timeout=timeout, connector=connector) as session:
            # Create tasks for all years
            tasks = [fetch_year_data(session, year) for year in years]

            # Execute concurrently
            completed_results = await asyncio.gather(*tasks, return_exceptions=True)

            # Process results
            for result in completed_results:
                if isinstance(result, Exception):
                    print(f"Task failed with exception: {result}")
                else:
                    year, data = result
                    results[year] = data

        return results

    def get_weekly_summary(self,
                           commodity_code: Optional[int] = None,
                           weeks: int = 52) -> pd.DataFrame:
        """
        Get weekly summary of export sales for the last N weeks

        Args:
            commodity_code (int, optional): Specific commodity to analyze
            weeks (int): Number of weeks to include (default: 52)

        Returns:
            pd.DataFrame: Weekly summary of export sales
        """
        end_date = datetime.now()
        start_date = end_date - pd.Timedelta(weeks=weeks)

        data = self.get_export_sales(
            commodity_code=commodity_code,
            start_date=start_date.date(),
            end_date=end_date.date()
        )

        if not data.empty and 'weekEndingDate' in data.columns:
            # Group by week and sum numeric columns
            data['week'] = data['weekEndingDate'].dt.to_period('W')
            numeric_cols = data.select_dtypes(include=['int64', 'float64']).columns
            weekly = data.groupby('week')[numeric_cols].sum().reset_index()
            weekly['week'] = weekly['week'].dt.start_time
            return weekly.sort_values('week')

        return data


# Example usage and utility functions
if __name__ == "__main__":
    # Initialize the API wrapper
    esr = USDAESR()

    # Example 1: Get all commodities
    print("Getting all commodities...")
    commodities = esr.get_commodities()
    print(f"Found {len(commodities)} commodities")
    print(commodities.head())

    # Example 2: Search for wheat commodities
    print("\nSearching for wheat commodities...")
    wheat_commodities = esr.search_commodities("wheat")
    print(wheat_commodities)

    # Example 3: Get all countries
    print("\nGetting all countries...")
    countries = esr.get_countries()
    print(f"Found {len(countries)} countries")

    # Example 4: Get export sales data (with error handling)
    try:
        print("\nGetting export sales data...")
        export_data = esr.get_export_sales(
            start_date='2023-01-01',
            end_date='2023-03-31'
        )
        print(f"Found {len(export_data)} export sales records")
        if not export_data.empty:
            print("\nColumns available:")
            print(export_data.columns.tolist())
            print("\nFirst few rows:")
            print(export_data.head())
    except USDAESRError as e:
        print(f"Error getting export sales data: {e}")

    # Example 5: Get weekly summary
    try:
        print("\nGetting weekly summary...")
        weekly = esr.get_weekly_summary(weeks=12)
        if not weekly.empty:
            print(f"Weekly summary for last 12 weeks:")
            print(weekly.head())
    except USDAESRError as e:
        print(f"Error getting weekly summary: {e}")
