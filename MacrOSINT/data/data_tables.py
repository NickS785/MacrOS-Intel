import datetime
import os
from pathlib import Path
import pandas as pd
import toml
from enum import Enum
import asyncio
from typing import List, Dict, Optional, Tuple, Union

# Google Drive integration removed

# Import from config instead of using dotenv directly
try:
    from MacrOSINT.config import (
        DATA_PATH, MARKET_DATA_PATH, COT_PATH,
        NASS_TOKEN, FAS_TOKEN, EIA_KEY,
        load_mapping, PROJECT_ROOT
    )
except ImportError:
    # Fallback for when config.py doesn't exist
    from dotenv import load_dotenv

    load_dotenv()
    DATA_PATH = os.getenv('data_path', './data')
    MARKET_DATA_PATH = os.getenv('market_data_path', './data/market')
    COT_PATH = os.getenv('cot_path', './data/cot')
    NASS_TOKEN = os.getenv('NASS_TOKEN', '')
    PROJECT_ROOT = Path(__file__).parent.parent

# Import external APIs
try:
    from myeia import API
    from MacrOSINT.data.sources.usda.api_wrappers.usda_quickstats import QuickStatsClient
    from CTAFlow.data import DataClient
except ImportError as e:
    print(f"Warning: Could not import external APIs: {e}")
    API = None
    QuickStatsClient = None

# Import local modules with proper paths
try:
    # Update import paths to match your structure
    from MacrOSINT.data.sources.usda.nass_utils import calc_dates, clean_numeric_column as clean_col, clean_fas
    from MacrOSINT.data.sources.usda.api_wrappers.psd_api import PSD_API, CommodityCode, CountryCode, AttributeCode, comms_dict, \
        rev_lookup, valid_codes, code_lookup
    from MacrOSINT.data.sources.usda.api_wrappers.esr_api import USDAESR
    from MacrOSINT.data.sources.usda.api_wrappers.esr_staging_api import USDAESRStaging
    from MacrOSINT.data.sources.usda.esr_csv_processor import ESRCSVProcessor, process_esr_csv
    from MacrOSINT.data.sources.eia.EIA_API import EIAClient, PetroleumClient, NaturalGasClient, EIAAPIError
    from MacrOSINT.data.sources.eia.api_tools import NatGasHelper as NGHelper, aggregate_regions, CENSUS_REGIONS, \
        PetroleumHelper as PHelper

except ImportError as e:
    print(f"Warning: Could not import USDA modules: {e}")
    # Provide None or mock objects
    calc_dates = None
    clean_col = None
    clean_fas = None
    PSD_API = None
    USDAESR = None
    USDAESRStaging = None
    ESRCSVProcessor = None
    process_esr_csv = None

# Import utilities
try:
    from MacrOSINT.utils.data_tools import walk_dict, key_to_name, convert_to_long

except ImportError:
    # Provide simple fallback implementations
    def walk_dict(d, parent_key=()):
        for k, v in d.items():
            full_key = parent_key + (k,)
            if isinstance(v, dict):
                yield from walk_dict(v, full_key)
            else:
                yield full_key, v


    def key_to_name(table_key):
        key_parts = table_key.rsplit('/')
        if len(key_parts) >= 2:
            name = f"{key_parts[-1]}_{key_parts[-2]}"
        else:
            # Single part key - just use the key itself
            name = key_parts[-1]
        return name


# Store folder
store_folder = Path(DATA_PATH)
store_folder.mkdir(parents=True, exist_ok=True)

# Load table mapping with proper path
try:
    mapping_path = PROJECT_ROOT / 'data' / 'sources' / 'eia' / 'data_mapping.toml'
    if mapping_path.exists():
        with open(mapping_path) as m:
            table_mapping = toml.load(m)
    else:
        print(f"Warning: EIA mapping file not found at {mapping_path}")
        table_mapping = {}
except Exception as e:
    print(f"Error loading table mapping: {e}")
    table_mapping = {}


# noinspection PyTypeChecker
class TableClient:
    """Base class for table data access"""

    def __init__(self, client, data_folder, db_file_name, key_prefix=None, map_file=None, api_data_col=None,
                 rename_on_load=False, verbose=False):
        self.client = client
        self.data_folder = Path(data_folder)
        self.table_db = Path(self.data_folder, db_file_name)
        self.data_col = api_data_col
        self.rename = rename_on_load
        self.app_path = PROJECT_ROOT
        self.verbose = verbose

        # Create data folder if it doesn't exist
        self.data_folder.mkdir(parents=True, exist_ok=True)

        if key_prefix is not None:
            self.prefix = key_prefix

        self.client_params = {}

        # Google Drive integration removed

        if map_file:
            try:
                with open(map_file) as m:
                    self.table_map_all = toml.load(m)
                self.mapping = self.table_map_all[self.prefix] if self.prefix else self.table_map_all
            except Exception as e:
                print(f"Error loading map file {map_file}: {e}")
                self.mapping = {}
        else:
            self.mapping = {}

    def available_keys(self, use_prefix=True):
        """Get available keys from HDF5 store"""
        try:
            with pd.HDFStore(self.table_db, mode='r') as store:
                all_keys = list(store.keys())
                print(f"Raw store keys: {all_keys}") if self.verbose else False  # Debug logging

                if hasattr(self, 'prefix') and self.prefix and use_prefix:
                    # Filter keys that start with the prefix
                    prefix_with_slash = f"/{self.prefix}/"
                    tables = []
                    for k in all_keys:
                        if k.startswith(prefix_with_slash):
                            # Remove the prefix and leading slash to get the relative key
                            relative_key = k[len(prefix_with_slash):]
                            tables.append(relative_key)
                    print(f"Filtered keys for prefix '{self.prefix}': {tables}")  # Debug logging
                else:
                    # Remove leading slash from all keys
                    tables = [k[1:] if k.startswith('/') else k for k in all_keys]
                    print(f"All keys (no prefix filter): {tables}")  # Debug logging

            return tables
        except Exception as e:
            print(f"Error accessing HDF5 store: {e}")
            return []

    def get_key(self, key, use_prefix=True, use_simple_name=True):
        """Get data for a specific key"""
        try:
            with pd.HDFStore(self.table_db, mode='r') as store:
                # Construct the full key path
                if hasattr(self, 'prefix') and self.prefix and use_prefix:
                    full_key = f'{self.prefix}/{key}'
                else:
                    full_key = key

                # Ensure key starts with '/' for HDF5 compatibility
                if not full_key.startswith('/'):
                    full_key = f'/{full_key}'
                print(f"Attempting to access key: {full_key}")  # Debug logging
                print(f"Available keys: {list(store.keys())}") if self.verbose else False # Debug logging

                table = store[full_key]

            col_value = table.columns[0] if self.data_col is None else self.data_col
            if use_simple_name:
                new_name = key_to_name(key)
                table = table.rename({col_value: new_name}, axis=1)

            return table
        except Exception as e:
            print(f"Error getting key {key} (full_key: {full_key if 'full_key' in locals() else 'unknown'}): {e}")

            return pd.DataFrame()

    def get_keys(self, keys: list, use_prefix=True, use_simple_name=True):
        """Get data for multiple keys"""
        dfs = []
        try:
            with pd.HDFStore(self.table_db, mode='r') as store:
                print(f"Available keys in store: {list(store.keys())}") if self.verbose else False # Debug logging

                for k in keys:
                    # Construct the full key path
                    if hasattr(self, 'prefix') and self.prefix and use_prefix:
                        full_key = f'{self.prefix}/{k}'
                    else:
                        full_key = k

                    # Ensure key starts with '/' for HDF5 compatibility
                    if not full_key.startswith('/'):
                        full_key = f'/{full_key}'

                    try:
                        print(f"Attempting to access key: {full_key}")  # Debug logging
                        table = store[full_key]
                        if isinstance(table, pd.DataFrame):
                            if use_simple_name:
                                col = key_to_name(k)
                                # Handle columns_col renaming logic
                                if len(table.columns) == 1:
                                    data = table.rename({table.columns[0]: col}, axis=1)
                                else:
                                    data = table
                        elif isinstance(table, pd.Series):
                            table.rename(key_to_name(k), axis=1)
                            data = table
                        else:
                            data = table

                        dfs.append(data)
                    except KeyError:
                        print(f"Key {full_key} not found in store")
                        continue

            if dfs:
                key_df = pd.concat(dfs, axis=1)
                return key_df
            else:
                return pd.DataFrame()

        except Exception as e:
            print(f"Error getting keys: {e}")
            return pd.DataFrame()

    def rename_table_keys(self, old_pattern: str, new_pattern: str, dry_run: bool = True) -> Dict[str, str]:
        """
        Rename tables in HDF5 store by pattern matching.
        
        Args:
            old_pattern: Pattern to match in existing keys (e.g., 'soybean')
            new_pattern: Replacement pattern (e.g., 'soybeans') 
            dry_run: If True, only show what would be renamed without making changes
            
        Returns:
            Dict mapping old keys to new keys
        """
        rename_map = {}

        try:
            with pd.HDFStore(self.table_db, mode='r') as store:
                all_keys = list(store.keys())

                # Find keys that match the pattern
                matching_keys = [k for k in all_keys if old_pattern in k]

                if not matching_keys:
                    print(f"No keys found matching pattern '{old_pattern}'")
                    return rename_map

                print(f"Found {len(matching_keys)} keys matching pattern '{old_pattern}':")
                for key in matching_keys:
                    new_key = key.replace(old_pattern, new_pattern)
                    rename_map[key] = new_key
                    print(f"  {key} -> {new_key}")

                if dry_run:
                    print("\nDry run mode - no changes made. Set dry_run=False to apply changes.")
                    return rename_map

            # Actually perform the rename operation
            print(f"\nRenaming {len(rename_map)} tables...")

            with pd.HDFStore(self.table_db, mode='a') as store:
                for old_key, new_key in rename_map.items():
                    try:
                        # Read data from old key
                        data = store[old_key]

                        # Write to new key
                        store[new_key] = data
                        print(f"✓ Copied {old_key} -> {new_key}")

                        # Remove old key
                        del store[old_key]
                        print(f"✓ Removed {old_key}")

                    except Exception as e:
                        print(f"✗ Error renaming {old_key} -> {new_key}: {e}")
                        # Try to clean up if new key was created but old key couldn't be deleted
                        try:
                            if new_key in store:
                                del store[new_key]
                        except:
                            pass

            print(f"\nCompleted renaming {len(rename_map)} tables")

        except Exception as e:
            print(f"Error during rename operation: {e}")

        return rename_map

    def list_all_tables(self) -> List[str]:
        """List all tables in the HDF5 store with their sizes."""
        try:
            with pd.HDFStore(self.table_db, mode='r') as store:
                all_keys = list(store.keys())

                print(f"HDF5 Store: {self.table_db}")
                print(f"Total tables: {len(all_keys)}")
                print("-" * 60)

                for key in sorted(all_keys):
                    try:
                        data = store[key]
                        if hasattr(data, 'shape'):
                            shape_info = f"{data.shape[0]} rows x {data.shape[1]} cols"
                        else:
                            shape_info = f"{len(data)} items"
                        print(f"{key:<40} {shape_info}")
                    except Exception as e:
                        print(f"{key:<40} Error: {e}")

                return all_keys

        except Exception as e:
            print(f"Error listing tables: {e}")
            return []

    def update_from_csv(self, key_name: str, csv_source: str, clean_function: object, clean_params=None,
                        read_csv_params=None, use_prefix=True, metadata=None):
        """

        :param key_name: key to save to table_db
        :param csv_source: string of csv location
        :param clean_function: function used to clean data, first argument should be for input data
        :param clean_params: parameters to be unpacked when cleaning
        :param read_csv_params: parameters of pd.read_csv
        :return: Bool
        """
        if read_csv_params is None:
            read_csv_params = {}
        if clean_params is None:
            clean_params = {}
        data = pd.read_csv(csv_source, **read_csv_params)

        table_data = clean_function(data, **clean_params)
        available_tables = {s.split('/')[0] for s in self.available_keys(use_prefix=False)}

        if self.prefix and use_prefix and self.prefix in available_tables:
            return self.update_table_data(f'{key_name}', data=table_data, use_prefix=use_prefix)

        elif key_name.split('/')[0] in available_tables:
            return self.update_table_data(f'{key_name}', data=table_data, use_prefix=False)
        else:
            print(f"Failed to detect a valid table for {key_name}")
            print(f"Is {self.prefix} or {key_name.split('/')[0]} in an existing table?")

        return

    def update_table_data(self, key, data, use_prefix=True, metadata=None):
        """
        Generic update method for storing data to any key in the HDF5 store.
        
        Args:
            key (str): Storage key for the data (e.g., 'usage/fsi', 'psd/summary')
            data (pd.DataFrame): Data to store
            use_prefix (bool): Whether to use the client's prefix for the key
            metadata (dict): Optional metadata to store with the data
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            if data is None or data.empty:
                print(f"Warning: No data provided for key {key}")
                return False

            # Construct the full key path
            if hasattr(self, 'prefix') and self.prefix and use_prefix:
                full_key = f'{self.prefix}/{key}'
            else:
                full_key = key

            # Ensure key starts with '/' for HDF5 compatibility
            if not full_key.startswith('/'):
                full_key = f'/{full_key}'

            # Store the data using HDF5
            data.to_hdf(self.table_db, key=full_key, mode='a', format='table', data_columns=True)

            print(f"Successfully stored {len(data)} rows to key: {full_key}")
            print(f"Database: {self.table_db}") if self.verbose else False

            # Store metadata if provided
            if metadata:
                metadata_key = f"{full_key}_metadata"
                try:
                    import json
                    metadata_df = pd.DataFrame([metadata])
                    metadata_df.to_hdf(self.table_db, key=metadata_key, mode='a', format='table')
                    print(f"Stored metadata to: {metadata_key}")
                except Exception as e:
                    print(f"Warning: Could not store metadata: {e}")

            return True

        except Exception as e:
            print(f"Error storing data to key {key}: {e}")
            import traceback
            traceback.print_exc()
            return False

    def delete_keys(self, keys:List):
        """ Function used to remove keys from table
            Arguments:
                keys : keys to be removed
            """
        with pd.HDFStore(self.table_db, 'a') as f:
            for k in keys:
                if k in f:
                    f.remove(k)
                    print(f'{k} successfully deleted')
                else:
                    print(f'Failed to find key: {k}')
        return

    def __getitem__(self, item):
        """Allow dict-like access"""
        if isinstance(item, list):
            return self.get_keys(item, use_prefix=True, use_simple_name=self.rename)
        elif isinstance(item, str):
            return self.get_key(item, use_prefix=True, use_simple_name=self.rename)

    # Google Drive Integration Methods removed


class MarketTable(DataClient):

    """Data access layer for market and COT data from HDF5 files."""

    def __init__(self):
        super().__init__()

        return

    def get_historical(self, ticker:Union[str,List], start_date='2015-01-01', end_date=None, resample_period:str=None, combine_datasets=False):
        """Get market data for a ticker."""
        daily = True if (resample_period.upper().endswith("D") or resample_period.upper().endswith("W")) else False
        historical_data = self.query_market_data(tickers=ticker, daily=daily, resample=resample_period, combine_datasets=combine_datasets)
        return historical_data

    def get_cot(self, commodity_symbol, type='metrics', start_date=None, end_date=None):
        """Get COT data for a commodity."""
        try:
            if type == 'metrics':
                metrics = self.query_cot_metrics(commodity_symbol, start_date=start_date, end_date=end_date)
                return metrics
            elif type == 'raw' or type == 'cot':
                raw_metrics = self.query_by_ticker(commodity_symbol, start_date=start_date, end_date=end_date)
                return raw_metrics

        except Exception as e:
            print(f"Error accessing COT data: {e}")
            return None

    def write_spot_data(self, symbol:str=None):
        if symbol.startswith("NG"):
            helper = NGHelper()
            is_petrol = False
        elif symbol[:2] in ["HO", "CL", "RB"]:
            helper = PHelper()
            is_petrol = True
        else:
            is_petrol = False
            raise ValueError("Unknown Symbol")

        data = helper.get_spot_prices()
        data = data.select_dtypes('float')

        if is_petrol:
            mapping = {"HO_F": ["EPD2DXL0", "EPJK"],
                       "RB_F": "EPMRU",
                       "CL_F": "EPCWTI"}

            for symbol_key, value in mapping.items():
                update_data = data[value]
                if isinstance(update_data, pd.Series):
                    update_data = update_data.to_frame(symbol_key)
                hdf_key = f"market/{symbol_key}/spot"
                self.write_market(update_data, hdf_key)
                print(f"Successful write to {hdf_key}")

        else:
            if (len(symbol) != 4) or not symbol.endswith('_F'):
                symbol_key = f"{symbol[:2]}_F"
            else:
                symbol_key = symbol
            if "units" in data.columns:
                data.drop(columns=["units"], inplace=True)

            hdf_key = f"market/{symbol_key}/spot"
            if isinstance(data, pd.Series):
                update_data = data.to_frame(symbol_key)
            else:
                update_data = data

            self.write_market(update_data, hdf_key)
            print(f"Successful write to {hdf_key}")

        return True


    def get_market_data(self, tickers, start_date=None, end_date=None, columns=None, 
                       resample=None, where=None, combine_datasets=False, daily=False):
        """
        Wrapper for query_market_data following TableClient methodology.
        
        Args:
            tickers: Ticker symbol(s) to query
            start_date: Start date filter in 'YYYY-MM-DD' format
            end_date: End date filter in 'YYYY-MM-DD' format  
            columns: Specific columns to return
            resample: Pandas resampling frequency
            where: Custom HDFStore where clause
            combine_datasets: If True, combines multiple tickers into single DataFrame
            daily: If True, queries daily resampled data
            
        Returns:
            DataFrame or dict of DataFrames with market data
        """
        try:
            return self.query_market_data(
                tickers=tickers,
                start_date=start_date,
                end_date=end_date,
                columns=columns,
                resample=resample,
                where=where,
                combine_datasets=combine_datasets,
                daily=daily
            )
        except Exception as e:
            print(f"Error accessing market data for {tickers}: {e}")
            return None

    def get_curve_data(self, symbol, curve_types=None, start_date=None, end_date=None,
                      columns=None, where=None, combine_datasets=False):
        """
        Wrapper for query_curve_data following TableClient methodology.
        
        Args:
            symbol: Base symbol (e.g., 'CL', 'ZC') 
            curve_types: Curve data types to query ('curve', 'volume_curve', 'oi_curve', etc.)
            start_date: Start date filter in 'YYYY-MM-DD' format
            end_date: End date filter in 'YYYY-MM-DD' format
            columns: Specific columns to return
            where: Custom HDFStore where clause
            combine_datasets: If True, combine all curve types into single DataFrame
            
        Returns:
            DataFrame or dict of DataFrames with curve data
        """
        try:
            return self.query_curve_data(
                symbol=symbol,
                curve_types=curve_types,
                start_date=start_date,
                end_date=end_date,
                columns=columns,
                where=where,
                combine_datasets=combine_datasets
            )
        except Exception as e:
            print(f"Error accessing curve data for {symbol}: {e}")
            return None

    def get_cot_by_ticker(self, tickers, **kwargs):
        """
        Wrapper for query_by_ticker following TableClient methodology.
        
        Args:
            tickers: Ticker symbol(s) to query
            **kwargs: Additional arguments passed to query_by_ticker
            
        Returns:
            DataFrame with COT data
        """
        try:
            return self.query_by_ticker(tickers, **kwargs)
        except Exception as e:
            print(f"Error accessing COT data by ticker {tickers}: {e}")
            return None

    def get_cot_metrics(self, ticker_symbol, columns=None, start_date=None, end_date=None, where=None):
        """
        Wrapper for query_cot_metrics following TableClient methodology.
        
        Args:
            ticker_symbol: The ticker symbol (e.g., 'ZC_F', 'CL_F')
            columns: Specific columns to retrieve
            start_date: Start date filter (YYYY-MM-DD format)
            end_date: End date filter (YYYY-MM-DD format)
            where: Additional where clause for filtering
            
        Returns:
            DataFrame with calculated COT metrics
        """
        try:
            return self.query_cot_metrics(
                ticker_symbol=ticker_symbol,
                columns=columns,
                start_date=start_date,
                end_date=end_date,
                where=where
            )
        except Exception as e:
            print(f"Error accessing COT metrics for {ticker_symbol}: {e}")
            return None


#############################################################################################################################################
## Energy Info tables
############################################################################################################################################

class EIATable(TableClient):
    """EIA data table client"""
    main_client = EIAClient()

    clients = {
        "NG": NGHelper(),
        "PET": PHelper()
    }

    def __init__(self, commodity, rename_key_cols=True):
        # Use the loaded mapping
        map_file = PROJECT_ROOT / 'data' / 'sources' / 'eia' / 'data_mapping.toml'

        super().__init__(
            self.clients[commodity].client,
            store_folder,
            'eia_data.h5',
            key_prefix=commodity,
            map_file=str(map_file),
            rename_on_load=rename_key_cols
        )

        self.commodity = self.prefix = commodity
        self.data_folder = Path(DATA_PATH) / 'EIA'


    def api_update_via_route(self, route: str, facet_params: dict, key: str, simple_col_names=True, use_prefix=True):
        """
        :param route_params:
        :param series:
        :return:
        """
        data_series = self.get_series_via_route(route, facet_params, key)

        self.update_table_data(data=data_series, key=key, use_prefix=use_prefix)

        return

    def get_series_via_route(self, route: str, facet_params: dict, start_date=None, end_date=None, **kwargs):
        date_params = {}
        if start_date:
            date_params.update({
                'start': start_date
            })
        if end_date:
            date_params.update({
                'end': end_date
            })

        return self.client.get_all_data(route, data_columns=["unit", "value"], facets=facet_params, **date_params)

    def update_overview(self, key_name="overview", use_prefix=True, folder_name=None, csv_name=None,
                        custom_function=None, custom_params=None):

        if use_prefix:
            file = self.data_folder / self.prefix.upper() / "overview.csv"
        else:
            if folder_name and csv_name:
                file = Path(folder_name) / csv_name
            else:
                "Folder and csv name required"
                raise FileExistsError

        if not custom_function:
            if self.update_from_csv(key_name, file, convert_to_long):
                print(f"Successfully updated {self.prefix}/{key_name}")
        else:
            if not custom_params:
                custom_params = {}
            self.update_from_csv(key_name, file, custom_function, clean_params=custom_params)

    def update_consumption_by_state(self, save=True, key="consumption/by_state", prefix=True, table=None):
        if self.prefix:
            df = self.clients["NG"].consumption_breakdown()
            full_key = f'{self.prefix}/{key}'
        elif table:
            df = self.clients[table]['Helper'].consumption_breakdown()
            full_key = f'{table}/{key}'
        else:
            print("Unable to locate commodity name")
            print(f"If a prefix is not used {table} must be given a value")
            raise KeyError

        if save:
            df.to_hdf(self.table_db, key=full_key)
            try:
                test_df = self.get_key(full_key, use_prefix=False, use_simple_name=False)
            except KeyError:
                raise KeyError
            else:
                if test_df.empty:
                    print(f"Saved key is empty")
                    return False
                else:
                    print(f"Key {full_key} Successfully saved!")

        return

    def create_update_dict(self, keys, dfs, funcs):
        update_dict = {}
        for key, df, func in zip(keys,dfs, funcs):
            update_dict[key] = {}
            update_dict[key]['df'] = df
            update_dict[key]['func'] = func

        return update_dict

    def get_consumption_by_region(self, start="2001-01-31", end="2025-05-31", end_use=None, update_table=False):
        """

        :param start: start date in format %yyyy-%mm-%dd
        :param end: date in format %yyyy-%mm-%dd
        :param end_use: Column to select to sum
        :return: dataframe
        """
        consumption_data = self.get_key('consumption/by_state', use_prefix=True, use_simple_name=False)
        # noinspection PyTypeChecker
        filtered_data = consumption_data[(consumption_data['date'] > start) & (consumption_data['date'] < end)]

        end_use = ["Delivered", "Electric Power", "Residential", "Commercial",
                   "Industrial"] if not end_use and self.prefix == "NG" else end_use
        df = aggregate_regions(filtered_data, sum_columns=end_use)
        if update_table:
            key = f'{self.prefix}/consumption/by_region'
            df.to_hdf(self.table_db, key=key)

        return df

    def update_petroleum_stocks(self, start="2001-01", end="2025-08", max_concurrent=3):
        """
        Update petroleum stocks using async methods for improved performance.
        
        Args:
            start: Start date in YYYY-MM format
            end: End date in YYYY-MM format
            max_concurrent: Maximum concurrent requests
        """
        import asyncio
        
        req_helper = self.clients['PET']
        
        async def _async_update():
            # Execute all requests concurrently using async methods
            refinery_df, refined_df, tank_farm_df = await asyncio.gather(
                req_helper.get_refinery_crude_stocks_async(start, end, max_concurrent),
                req_helper.get_product_stocks_async(start, end, max_concurrent),
                req_helper.get_tank_farm_stocks_async(start, end, max_concurrent),
                return_exceptions=True
            )
            return refinery_df, refined_df, tank_farm_df
        
        # Run the async operations
        dfs_to_write = []
        try:
            refinery_df, refined_df, tank_farm_df = asyncio.run(_async_update())
            
            # Save the results if successful
            if not isinstance(tank_farm_df, Exception):
                dfs_to_write.append({'key':"PET/supply/tank_crude_stocks", 'df':tank_farm_df})
                print(f"Saved tank farm stocks: {len(tank_farm_df)} records")
            else:
                print(f"Failed to update tank farm stocks: {tank_farm_df}")
                
            if not isinstance(refinery_df, Exception):
                dfs_to_write.append({'key':"PET/refineries/crude_stocks", 'df':refinery_df})
                print(f"Saved refinery crude stocks: {len(refinery_df)} records")
            else:
                print(f"Failed to update refinery crude stocks: {refinery_df}")
                
            if not isinstance(refined_df, Exception):
                dfs_to_write.append({'key':"PET/supply/refined_stocks", 'df':refined_df})
                print(f"Saved refined product stocks: {len(refined_df)} records")
            else:
                print(f"Failed to update refined product stocks: {refined_df}")
                
        except Exception as e:
            print(f"Failed to update petroleum stocks: {e}")
            # Fallback to sync methods if async fails
            print("Falling back to synchronous methods...")
            try:
                update_df = req_helper.get_refinery_crude_stocks(start, end)
                refined_update_df = req_helper.get_product_stocks(start, end)
                tank_farm_update_df = req_helper.get_tank_farm_stocks(start, end)

                tank_farm_update_df.to_hdf(self.table_db, "PET/supply/tank_crude_stocks")
                update_df.to_hdf(self.table_db, "PET/refineries/crude_stocks")
                refined_update_df.to_hdf(self.table_db, "PET/supply/refined_stocks")
                print("Successfully updated using synchronous methods")
            except Exception as sync_error:
                print(f"Synchronous fallback also failed: {sync_error}")
        if dfs_to_write:
            with pd.HDFStore(self.table_db) as tdb:
                for entry in dfs_to_write:
                    tdb.put(key=entry['key'], value=entry['df'])
                    print(f'Successfully stored {entry["key"]} in {self.table_db}')


                raise

    async def update_petroleum_stocks_async(self, start="2001-01", end="2025-08", max_concurrent=3):
        """
        Async version of update_petroleum_stocks for improved performance.
        
        Args:
            start: Start date in YYYY-MM format
            end: End date in YYYY-MM format  
            max_concurrent: Maximum concurrent requests
            
        Returns:
            Dictionary with update results for each stock type
        """
        import asyncio
        
        req_helper = self.clients['PET']
        
        # Execute all requests concurrently using async methods
        try:
            # Use asyncio.gather to run all async tasks concurrently
            refinery_df, refined_df, tank_farm_df = await asyncio.gather(
                req_helper.get_refinery_crude_stocks_async(start, end, max_concurrent),
                req_helper.get_product_stocks_async(start, end, max_concurrent),
                req_helper.get_tank_farm_stocks_async(start, end, max_concurrent),
                return_exceptions=True
            )
            
            results = {}
            
            # Save refinery crude stocks
            if not isinstance(refinery_df, Exception):
                refinery_df.to_hdf(self.table_db, "PET/refineries/crude_stocks")
                results['refinery_crude'] = {
                    'success': True, 
                    'records': len(refinery_df),
                    'key': 'PET/refineries/crude_stocks'
                }
                print(f"Saved refinery crude stocks: {len(refinery_df)} records")
            else:
                results['refinery_crude'] = {
                    'success': False, 
                    'error': str(refinery_df)
                }
                print(f"Failed to update refinery crude stocks: {refinery_df}")
            
            # Save refined product stocks  
            if not isinstance(refined_df, Exception):
                refined_df.to_hdf(self.table_db, "PET/supply/refined_stocks")
                results['refined_products'] = {
                    'success': True,
                    'records': len(refined_df),
                    'key': 'PET/supply/product_stocks'
                }
                print(f"Saved refined product stocks: {len(refined_df)} records")
            else:
                results['refined_products'] = {
                    'success': False,
                    'error': str(refined_df)
                }
                print(f"Failed to update refined product stocks: {refined_df}")
            
            # Save tank farm stocks
            if not isinstance(tank_farm_df, Exception):
                tank_farm_df.to_hdf(self.table_db, "PET/supply/tank_crude_stocks")
                results['tank_farm'] = {
                    'success': True,
                    'records': len(tank_farm_df), 
                    'key': 'PET/supply/tank_crude_stocks'
                }
                print(f"Saved tank farm stocks: {len(tank_farm_df)} records")
            else:
                results['tank_farm'] = {
                    'success': False,
                    'error': str(tank_farm_df)
                }
                print(f"Failed to update tank farm stocks: {tank_farm_df}")
            
            return results
            
        except Exception as e:
            print(f"Async petroleum stocks update failed: {e}")
            return {
                'refinery_crude': {'success': False, 'error': str(e)},
                'refined_products': {'success': False, 'error': str(e)},
                'tank_farm': {'success': False, 'error': str(e)}
            }

    def update_petroleum_production(self, start=None, end=None):
        req_helper = self.clients['PET']
        if not start:
            start = "2001-01"
        if not end:
            end = "2025-08"

        update_crude = req_helper.get_production_by_area(start, end)
        update_refined = req_helper.get_refined_products_production(start, end)

        update_data = {'PET/production/refined_products': {'df': update_refined, 'func':req_helper.get_refined_products_production},
         'PET/production/crude_oil': {'df': update_crude, 'func': req_helper.get_production_by_area}}

        self.update_from_dict(start, end, update_data)

        return

    def update_petroleum_consumption(self, start=None, end=None):
        if not start:
            start = "2001-01"
        if not end:
            end = "2025-08"
        helper = self.clients['PET']
        update_consumption =  {'/PET/consumption/by_product': {'df':helper.us_weekly_consumption_breakdown(start, end), 'func': helper.get_product_supplied}}

        self.update_from_dict(start, end, update_consumption)

        return

    def update_petroleum_movements(self, start=None, end=None):
        """
        Updating intra-district movements.
        movements/ is net, movements/producers is flow between refining/ producing / importing / exporting states
        """
        if not start:
            start = "2001-01"
        if not end:
            end = "2025-08"
        helper = self.clients['PET']

        net_receipts_df = helper.get_product_movements(start, end)
        crude_movements_df = helper.get_crude_movements(start, end)
        update_dict = {'/PET/movements/net':{'df':net_receipts_df, 'func':helper.get_product_movements},
                      '/PET/movements/crude_movements':{'df':crude_movements_df, 'func':helper.get_crude_movements}}

        self.update_from_dict(start, end, update_dict)

        return

    async def update_petroleum_movements_async(self, start=None, end=None, max_concurrent=3, progress_callback=None):
        """
        Async version of update_petroleum_movements for improved performance with concurrent requests.
        
        Args:
            start: Start date in format "YYYY-MM" (default: "2001-01")
            end: End date in format "YYYY-MM" (default: "2025-08")
            max_concurrent: Maximum concurrent requests (default: 3)
            progress_callback: Optional callback function for progress updates
            
        Returns:
            dict: Results with successful/failed updates and timing info
        """
        if not start:
            start = "2001-01"
        if not end:
            end = "2025-08"
            
        helper = self.clients['PET']
        
        try:
            print(f"Starting async petroleum movements update for period {start} to {end}")
            start_time = pd.Timestamp.now()
            
            results = {
                'successful_data': {},
                'failed_data': {},
                'stored_keys': [],
                'errors': {}
            }
            
            # Get net movements using existing sync method (if no async version exists)
            try:
                net_receipts_df = helper.get_product_movements(start, end)
                if not net_receipts_df.empty:
                    results['successful_data']['net_movements'] = net_receipts_df
                    net_receipts_df.to_hdf(self.table_db, '/PET/movements/net')
                    results['stored_keys'].append('/PET/movements/net')
                    print(f"✓ Stored {len(net_receipts_df)} rows to PET/movements/net")
                else:
                    results['failed_data']['net_movements'] = "No data returned"
                    
            except Exception as e:
                error_msg = f"Failed to get net movements data: {str(e)}"
                results['failed_data']['net_movements'] = error_msg
                results['errors']['net_movements'] = error_msg
                print(f"✗ {error_msg}")
            
            # Get crude movements using async method
            try:
                crude_movements_df = await helper.get_crude_movements_async(
                    start=start,
                    end=end,
                    max_concurrent=max_concurrent,
                    progress_callback=progress_callback
                )
                
                if not crude_movements_df.empty:
                    results['successful_data']['crude_movements'] = crude_movements_df
                    # Store to HDF5 with MultiIndex columns preserved
                    crude_movements_df.to_hdf(self.table_db, '/PET/movements/crude_movements')
                    results['stored_keys'].append('/PET/movements/crude_movements')
                    print(f"✓ Stored {len(crude_movements_df)} rows to PET/movements/crude_movements")
                else:
                    results['failed_data']['crude_movements'] = "No data returned"
                    
            except Exception as e:
                error_msg = f"Failed to get crude movements data: {str(e)}"
                results['failed_data']['crude_movements'] = error_msg
                results['errors']['crude_movements'] = error_msg
                print(f"✗ {error_msg}")
            
            end_time = pd.Timestamp.now()
            duration = (end_time - start_time).total_seconds()
            
            # Update results with timing info
            results['duration_seconds'] = duration
            results['start_time'] = start_time
            results['end_time'] = end_time
            
            print(f"Async petroleum movements update completed in {duration:.2f} seconds")
            print(f"Successfully stored {len(results['stored_keys'])} datasets: {results['stored_keys']}")
            
            if results.get('failed_data'):
                print(f"Failed to retrieve: {list(results['failed_data'].keys())}")
                
            return results
            
        except Exception as e:
            error_msg = f"Critical error in async petroleum movements update: {str(e)}"
            print(f"💥 {error_msg}")
            return {
                'error': error_msg,
                'successful_data': {},
                'failed_data': {'critical': error_msg},
                'stored_keys': [],
                'duration_seconds': 0
            }

    def update_petroleum_movements_async_sync(self, start=None, end=None, max_concurrent=3, progress_callback=None):
        """
        Synchronous wrapper for the async petroleum movements update method.
        
        Args:
            start: Start date in format "YYYY-MM" (default: "2001-01")
            end: End date in format "YYYY-MM" (default: "2025-08")
            max_concurrent: Maximum concurrent requests (default: 3)
            progress_callback: Optional callback function for progress updates
            
        Returns:
            dict: Results from async operation
        """
        # Check if we're already in an event loop
        try:
            loop = asyncio.get_running_loop()
            # We're already in an async context, need to create a new task
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor() as executor:
                future = executor.submit(
                    asyncio.run,
                    self.update_petroleum_movements_async(start, end, max_concurrent, progress_callback)
                )
                return future.result()
        except RuntimeError:
            # No running loop, safe to use asyncio.run
            return asyncio.run(
                self.update_petroleum_movements_async(start, end, max_concurrent, progress_callback)
            )

    def update_weekly_trade(self, start=None, end=None):
        """ Function to update import/export tables """
        if not start:
            start = "2001-01"
        if not end:
            end = "2025-08"

        helper = self.clients['PET']

        update_dict = {
                        'PET/movements/imports':
                           {
                            'df':helper.imports_weekly(start=start, end=end),
                            'func': helper.imports_weekly
                           },
                        'PET/movements/exports':
                           {
                               'df': helper.exports_weekly(start=start, end=end),
                               'func': helper.exports_weekly}
                        }

    def update_monthly_trade(self, start=None, end=None):
        if not start:
            start = "2001-01"
        if not end:
            end = "2025-08"

        update_func = lambda x, y: helper.get_padd_imports_from_top_src(padd_codes=["R10", 'R20', 'R30', "R40", "R50"], start=x, end=y)

        helper = self.clients['PET']
        update_dict = {
        'PET/movements/exports/by_district':
            {
            'df': helper.get_padd_exports(start, end),
            'func': helper.get_padd_exports
            },
        'PET/movements/imports/by_district':
            {
            'df': helper.get_padd_imports_from_top_src(start=start, end=end),
            'func': update_func
            }
        }


        self.update_from_dict(start, end, update_dict)


        return

    async def update_monthly_trade_async(self, start=None, end=None, max_concurrent=3, progress_callback=None):
        """
        Async version of update_monthly_trade using async EIA client for improved performance.
        
        Args:
            start: Start date in format "YYYY-MM" (default: "2001-01")
            end: End date in format "YYYY-MM" (default: "2025-08") 
            max_concurrent: Maximum concurrent requests (default: 3)
            progress_callback: Optional callback function for progress updates
            
        Returns:
            dict: Results with successful/failed updates and timing info
        """
        if not start:
            start = "2001-01"
        if not end:
            end = "2025-08"

        helper = self.clients['PET']
        
        try:
            print(f"Starting async monthly trade update for period {start} to {end}")
            start_time = pd.Timestamp.now()
            
            # Use async methods from PetroleumHelper
            results = {
                'successful_data': {},
                'failed_data': {},
                'stored_keys': [],
                'errors': {}
            }
            
            # Get imports using async method
            try:
                padd_codes = ["R10", "R20", "R30", "R40", "R50"]
                imports_data = await helper.get_multiple_padd_imports_async(
                    padd_codes=padd_codes,
                    start=start,
                    end=end,
                    max_concurrent=max_concurrent,
                    progress_callback=progress_callback)

                
                if not imports_data.empty:
                    results['successful_data']['imports_by_district'] = imports_data
                    
                    # Ensure columns are flattened for HDF5 compatibility
                    if isinstance(imports_data.columns, pd.MultiIndex):
                        # Flatten MultiIndex columns 
                        imports_data.columns = ['_'.join(str(level) for level in col if level != '').strip('_') 
                                              for col in imports_data.columns]
                    
                    # Store to HDF5
                    imports_data.to_hdf(self.table_db, 'PET/movements/imports/by_district')
                    results['stored_keys'].append('PET/movements/imports/by_district')
                    print(f"✓ Stored {len(imports_data)} rows to PET/movements/imports/by_district")
                else:
                    results['failed_data']['imports_by_district'] = "No data returned"
                    
            except Exception as e:
                error_msg = f"Failed to get imports data: {str(e)}"
                results['failed_data']['imports_by_district'] = error_msg
                results['errors']['imports'] = error_msg
                print(f"✗ {error_msg}")
            
            # Get exports using async method
            try:
                exports_data = await helper.get_exports_async(
                    start=start,
                    end=end,
                    max_concurrent=max_concurrent
                )
                
                if not exports_data.empty:
                    results['successful_data']['exports_by_district'] = exports_data
                    # Store to HDF5 with MultiIndex columns preserved
                    exports_data.to_hdf(self.table_db, 'PET/movements/exports/by_district')
                    results['stored_keys'].append('PET/movements/exports/by_district')
                    print(f"✓ Stored {len(exports_data)} rows to PET/movements/exports/by_district")
                else:
                    results['failed_data']['exports_by_district'] = "No data returned"
                    
            except Exception as e:
                error_msg = f"Failed to get exports data: {str(e)}"
                results['failed_data']['exports_by_district'] = error_msg
                results['errors']['exports'] = error_msg
                print(f"✗ {error_msg}")
            
            end_time = pd.Timestamp.now()
            duration = (end_time - start_time).total_seconds()
            
            # Update results with timing info
            results['duration_seconds'] = duration
            results['start_time'] = start_time
            results['end_time'] = end_time
            
            print(f"Async monthly trade update completed in {duration:.2f} seconds")
            print(f"Successfully stored {len(results['stored_keys'])} datasets: {results['stored_keys']}")
            
            if results.get('failed_data'):
                print(f"Failed to retrieve: {list(results['failed_data'].keys())}")
                
            return results
            
        except Exception as e:
            error_msg = f"Critical error in async monthly trade update: {str(e)}"
            print(f"💥 {error_msg}")
            return {
                'error': error_msg,
                'successful_data': {},
                'failed_data': {'critical': error_msg},
                'stored_keys': [],
                'duration_seconds': 0
            }

    def update_monthly_trade_async_sync(self, start=None, end=None, max_concurrent=3, progress_callback=None):
        """
        Synchronous wrapper for the async monthly trade update method.
        
        Args:
            start: Start date in format "YYYY-MM" (default: "2001-01")
            end: End date in format "YYYY-MM" (default: "2025-08")
            max_concurrent: Maximum concurrent requests (default: 3)
            progress_callback: Optional callback function for progress updates
            
        Returns:
            dict: Results from async operation
        """
        # Check if we're already in an event loop
        try:
            loop = asyncio.get_running_loop()
            # We're already in an async context, need to create a new task
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor() as executor:
                future = executor.submit(
                    asyncio.run,
                    self.update_monthly_trade_async(start, end, max_concurrent, progress_callback)
                )
                return future.result()
        except RuntimeError:
            # No running loop, safe to use asyncio.run
            return asyncio.run(
                self.update_monthly_trade_async(start, end, max_concurrent, progress_callback)
            )

    def update_from_dict(self,  start:str,end:str, update_dict:dict):

        """ Formula used to update hdf without empty keys and to retry failed connections
            Args:
                end (str): "YYYY-MM" formatted string
                start (str): "YYYY-MM" formatted string
                update_dict (dict) : Dict of dicts structured {hdf_key:{'df':update_dataframe (pd.DataFrame), 'func':retrieval_function (func) }
        """


        failed_keys = []
        # Retry once if failed
        for key, val_dict in update_dict.items():
            df = val_dict['df']
            retrieval_func = val_dict['func']

            # Only updates if df is not empty
            if df.empty:
                try:
                    # noinspection PyArgumentList
                    update_df = retrieval_func(start=start, end=end)
                except EIAAPIError as e:
                    print(f"Failed to retrieve client data {e}")
                    failed_keys.append(key)
                else:
                    if not update_df.empty:
                        update_df.to_hdf(self.table_db, key)
                    else:
                        print(f"DataFrame retrieved was empty failure to update key : {key}")
            else:
                df.to_hdf(self.table_db, key)

        if failed_keys:
             print(f'Failed to update keys : {failed_keys}')

        return

    def update_refinery_data(self, start=None, end=None):
        if not start:
            start = "2001-01-01"
        if not end:
            end = "2025-08-15"

        req_helper = self.clients['PET']

        try:
            refinery_utilization = req_helper.get_refinery_utilization(start, end)
            refinery_crude_stocks = req_helper.get_refinery_crude_stocks(start, end)
            refinery_crude_consumption = req_helper.get_refinery_consumption(start, end)
            refinery_output = req_helper.get_refined_products_production(start, end)
        except EIAAPIError as e:
            print('Failed to retrieve data from server\n')
            print(f'Error generated: {e}')
            return
        else:
            pass

        refinery_utilization.to_hdf(self.table_db, 'PET/refineries/utilization')
        refinery_crude_stocks.to_hdf(self.table_db, 'PET/refineries/crude_stocks')
        refinery_crude_consumption.to_hdf(self.table_db, 'PET/refineries/crude_consumption')
        refinery_output.to_hdf(self.table_db, "PET/production/refined_products")

        print("Successfully updated all refinery data")
        return True



#############################################################################################################################################
##
##
## Agricultural tables
############################################################################################################################################
# Initialize QuickStats client

if QuickStatsClient:
    _quickstats_client = QuickStatsClient()
else:
    _quickstats_client = None

nass_info = {
    'key': os.getenv('NASS_TOKEN'),
    'client': lambda x: _quickstats_client.query_df_numeric(**x) if _quickstats_client else None}


# National Agricultural Stats
class NASSTable(TableClient):
    with open(Path(PROJECT_ROOT, 'data', 'sources', 'usda', 'data_mapping.toml')) as map_file:
        table_map = toml.load(map_file)

    def __init__(self,
                 prefix=None):
        self.prefix = self.table = prefix
        super().__init__(nass_info['client'], data_folder=store_folder, db_file_name="nass_agri_stats.hd5",
                         api_data_col='Value', key_prefix=prefix, rename_on_load=True)

        self.mapping = self.table_map[prefix] if prefix else self.table_map

        return

    def get_descs(self,
                  param_value='short_desc'):

        if _quickstats_client is None:
            raise RuntimeError("QuickStatsClient not available")
        return _quickstats_client.get_param_values(param_value)

    def api_update(self, short_desc,
                   key=None,
                   commodity_desc=None,
                   freq_desc_preference='MONTHLY',
                   agg_level=None,
                   year_ge: str = None,
                   year_lt: str = None,
                   state=None,
                   county=None,
                   year=None,
                   use_prefix=False, **filters):
        ''' Used to write tables to the store using USDA's NASS API '''
        agg_level = 'NATIONAL' if agg_level is None else agg_level

        if not commodity_desc:
            commodity_desc = short_desc.split(' ')[0]
        if commodity_desc not in ["CATTLE", "BEEF", "HOGS", "PORK"]:
            sector_desc = "CROPS"
            group_desc = "FIELD CROPS"
        else:
            sector_desc, group_desc = "ANIMALS & PRODUCTS", "LIVESTOCK"

        params = {}
        params.update({
            "source_desc": "SURVEY",
            'sector_desc': sector_desc.upper(),
            'group_desc': group_desc.upper(),
            "commodity_desc": commodity_desc.upper(),
            'agg_level_desc': agg_level.upper(),
            'short_desc': short_desc.upper(),
            'domain_desc': 'TOTAL',

        })

        if state is not None:
            params.update({
                'state_name': state,
            })
        if county is not None:
            params.update({
                'county_name': county
            })
        if year and not year_ge:
            if isinstance(year, str):
                params.update({
                    'year': year
                })
        if year_ge:
            params.update({
                'year__GE': year_ge
            })
        if year_lt:
            params.update({
                'year__LT': year_lt
            })

        try:
            # Debug: Print parameters being passed to API
            print(f"DEBUG api_update() params: {params}")
            # Use the lambda client function which returns a DataFrame
            df = self.client(params)

        except Exception as e:
            print(
                f'Series {short_desc} not able to be downloaded\n Exception {e}'
            )
            return
        else:
            # df is already a DataFrame from query_df_numeric
            if 'freq_desc' in df.columns:
                freqs = df['freq_desc'].unique().tolist()
                if len(freqs) > 1:
                    if freq_desc_preference in freqs:
                        df = df.loc[df['freq_desc'].str.upper() == freq_desc_preference.upper()]
                    else:
                        # Save the frequency with most datapoints
                        biggest_len = 0
                        largest_fq = None
                        for fq in freqs:
                            fq_len = len(df.loc[df['freq_desc'] == fq])
                            biggest_len, largest_fq = (fq_len, fq) if fq_len > biggest_len else (
                                biggest_len, largest_fq)
                        if largest_fq == 'POINT IN TIME':
                            # Renamed for locating values
                            df = df.loc[df['freq_desc'] == largest_fq]
                            largest_fq = 'MONTHLY'
                            df.loc[:, 'freq_desc'] = largest_fq

                # Try to calculate dates using the utility function
                date_calculation_success = False
                try:
                    if calc_dates is not None:
                        df_dates = calc_dates(df)
                        df.index = df_dates
                        df['date'] = df.index
                        date_calculation_success = True
                        if key is None:
                            key = f'{short_desc.lower()}'
                    else:
                        print(f'calc_dates utility not available for {short_desc}')

                except Exception as e:
                    print(f'Series {short_desc} failed at calculating datetimes using calc_dates utility')
                    print(f'Error: {e}')
                    print(f'Columns available: {list(df.columns)}')

                # Fallback date handling if calc_dates failed

                if not date_calculation_success:
                    try:
                        # Try basic date construction based on available columns
                        if 'year' in df.columns and 'end_code' in df.columns:
                            # Filter out invalid end codes
                            df = df[df['end_code'].astype(str).str.isnumeric()]
                            df = df[(df['end_code'].astype(int) >= 1) & (df['end_code'].astype(int) <= 12)]

                            # Create date using year and month (end_code)
                            df['date'] = df['year'].astype(str) + '-' + df['end_code'].astype(str).str.zfill(2) + '-28'
                            df['date'] = pd.to_datetime(df['date'], errors='coerce')
                            df.index = df['date']
                            date_calculation_success = True
                            print(f'Used fallback date calculation for {short_desc}')

                        elif 'year' in df.columns:
                            # If only year is available, use December 31st
                            df['date'] = df['year'].astype(str) + '-12-31'
                            df['date'] = pd.to_datetime(df['date'], errors='coerce')
                            df.index = df['date']
                            date_calculation_success = True
                            print(f'Used year-only date calculation for {short_desc}')

                    except Exception as fallback_error:
                        print(f'Fallback date calculation also failed for {short_desc}: {fallback_error}')

                # If all date calculations failed, create a simple index
                if not date_calculation_success:
                    print(f'Warning: No date columns_col created for {short_desc} - using row index')
                    # Don't create a date columns_col, but still allow data storage
                    if key is None:
                        key = f'{short_desc.lower()}'

            else:
                # Handle case where there's no freq_desc columns_col
                try:
                    if 'year' in df.columns and 'end_code' in df.columns:
                        # Filter and create dates
                        df = df[df['end_code'].astype(str).str.isnumeric()]
                        df = df[(df['end_code'].astype(int) >= 1) & (df['end_code'].astype(int) <= 12)]
                        df['date'] = df['year'].astype(str) + '-' + df['end_code'].astype(str).str.zfill(2) + '-28'
                        df['date'] = pd.to_datetime(df['date'], errors='coerce')
                        df.index = df['date']
                    else:
                        print(f'No standard date columns found for {short_desc}')
                except Exception as e:
                    print(f'Error in non-freq_desc date processing for {short_desc}: {e}')

            # Store the data if we have a key and some data
            if key is not None and not df.empty:
                try:
                    df.sort_index(inplace=True)
                    df[key_to_name(key)] = clean_col(df['Value'])

                    if use_prefix:
                        table_key = f'{self.prefix}/{key}'
                    else:
                        table_key = key

                    # Prepare columns for storage - only include columns that exist
                    storage_columns = []
                    if 'date' in df.columns:
                        storage_columns.append('date')
                    if 'commodity_desc' in df.columns:
                        storage_columns.append('commodity_desc')
                    # Add essential state-level columns for non-national data
                    if agg_level != "NATIONAL" or filters.get('agg_level_desc', 'NATIONAL') != 'NATIONAL':
                        if 'state_alpha' in df.columns:
                            storage_columns.append("state_alpha")
                        if 'state_fips_code' in df.columns:
                            storage_columns.append("state_fips_code")
                        if 'state_ansi' in df.columns:
                            storage_columns.append("state_ansi")
                        if 'state_name' in df.columns:
                            storage_columns.append("state_name")

                    # Add essential county-level columns including FIPS codes
                    if agg_level == "COUNTY":
                        if 'county_name' in df.columns:
                            storage_columns.append("county_name")
                        if 'county_code' in df.columns:
                            storage_columns.append("county_code")
                        if 'county_ansi' in df.columns:
                            storage_columns.append("county_ansi")
                        if 'zip_5' in df.columns:
                            storage_columns.append("zip_5")
                    if agg_level == "COUNTY" or agg_level == "AGRICULTURAL REGION":
                        if 'location_desc' in df.columns:
                            storage_columns.append("location_desc")

                    storage_columns.append(key_to_name(key))  # This is the processed value columns_col

                    if 'year' in df.columns:
                        storage_columns.append('year')
                    if 'end_code' in df.columns:
                        storage_columns.append('end_code')

                    # Store only the columns that exist
                    df[storage_columns].to_hdf(self.table_db, key=f'{table_key}')
                    print(f'USDA Statistic {short_desc} saved to {Path(self.table_db)} in key {table_key}')
                    return True

                except Exception as storage_error:
                    print(f'Error storing {short_desc}: {storage_error}')
                    return False
            else:
                print(f'No data to store for {short_desc}')
                return False

    def api_update_all(self):
        failed = {}
        for k, v in walk_dict(self.mapping):
            table_key = f''
            for i in range(len(k)):
                table_key = table_key + '/' + k[i]
            try:
                self.api_update(short_desc=v, key=table_key)

            except Exception as e:
                print(f"Exception Raised with the following error: {e}")
                failed.update({k: v})
            else:
                print(self[f'{table_key}'])

        if failed:
            print(f'The following keys failed to update: {failed}')
            return failed
        else:
            print('All keys were successfully updated')

        return

    def api_update_commodity(self, commodity, **api_params):
        series_map = self.table_map[commodity]
        original_keys = []
        successful_keys = []
        for multi_key, short_desc in walk_dict(series_map):
            # Create the table key from tuple
            table_key = f'/{commodity}'
            for i in range(len(multi_key)):
                table_key = table_key + '/' + multi_key[i]

            original_keys.append(table_key)
            table_key = table_key[:-1] if table_key.endswith('/') else table_key
            if self.api_update(short_desc, key=table_key, **api_params):
                successful_keys.append(table_key)

        missed_keys = [*set(original_keys).difference(successful_keys)]

        return print(
            f'{len(successful_keys)}/{len(original_keys)} Successfully downloaded.\n\n Keys {missed_keys} failed to download')

    def _get_acres_planted(self, commodity_desc: str, agg_level='COUNTY', update_keys=False, year=None, start_year=2010,
                           end_year=2025):
        """
        Fetch Acres Planted for an area
        Args:
            year : calendar year acres planted
            commodity_desc: selected commodity
            agg_level: Aggregation level of data
            update_keys: Set to True if you wish to save the data to self.table_db
        """
        if isinstance(start_year, int):
            year_ge = str(start_year)
            year_lt = str(end_year)
        if not year:
            date_params = {"year__GE": start_year, "year__LE": end_year}
        else:
            if isinstance(year, int):
                year = str(year)

            date_params = {"year": year}

        req_params = {
            "source_desc": "SURVEY",
            "sector_desc": "CROPS",
            "group_desc": "FIELD CROPS",
            "commodity_desc": commodity_desc.upper(),
            "short_desc": f"{commodity_desc.upper()} - ACRES PLANTED",
            "domain_desc": "TOTAL",
            "agg_level_desc": agg_level,
            **date_params
        }
        # Debug: Print parameters being passed to API
        print(f"DEBUG _get_acres_planted() params: {req_params}")
        acres_df = self.client(req_params)
        if isinstance(acres_df, str):
            print("Failed to retrieve JSON from server, likely bad parameters")
            print(acres_df)
            return False
        else:
            if update_keys:
                acres_df.to_hdf(self.table_db,
                                key=f"/{commodity_desc.lower()}/production/acres_planted/{start_year}_{end_year}")

        return acres_df

    async def api_update_multi_year_async(self, short_desc: str, start_year: int, end_year: int,
                                          commodity_desc: str = None, agg_level: str = 'NATIONAL',
                                          state: str = None, county: str = None, max_concurrent: int = 3,
                                          freq_desc_preference: str = 'MONTHLY', key_desc=None) -> Dict[str, any]:
        """
        Update NASS data for multiple years using async requests for speed.
        
        Args:
            short_desc: NASS short description (e.g., 'CORN - ACRES PLANTED')
            start_year: Starting year (inclusive)
            end_year: Ending year (inclusive)
            commodity_desc: Commodity description (auto-detected if None)
            agg_level: Aggregation level ('NATIONAL', 'STATE', 'COUNTY')
            state: State name filter (optional)
            county: County name filter (optional)
            max_concurrent: Maximum concurrent requests (default: 3)
            freq_desc_preference: Preferred frequency for data (default: 'MONTHLY')
            
        Returns:
            Dict with update results and summary statistics
        """
        if not _quickstats_client:
            raise RuntimeError("QuickStatsClient not available")

        if not commodity_desc:
            commodity_desc = short_desc.split(' ')[0]

        key_part = f'{commodity_desc.lower()}/{key_desc}'

        years = list(range(start_year, end_year + 1))
        results = {
            'successful_years': [],
            'failed_years': [],
            'errors': {},
            'updated_tables': [],
            'summary': {}
        }

        # Determine sector and group like api_update method
        if commodity_desc not in ["CATTLE", "BEEF", "HOGS", "PORK"]:
            sector_desc = "CROPS"
            group_desc = "FIELD CROPS"
        else:
            sector_desc, group_desc = "ANIMALS & PRODUCTS", "LIVESTOCK"

        # Base parameters that are common to all requests (match api_update format)
        base_params = {
            "source_desc": "SURVEY",
            'sector_desc': sector_desc.upper(),
            'group_desc': group_desc.upper(),
            "commodity_desc": commodity_desc.upper(),
            'agg_level_desc': agg_level.upper(),
            'short_desc': short_desc.upper(),
            'domain_desc': 'TOTAL',
        }

        if state:
            base_params['state_name'] = state
        if county:
            base_params['county_name'] = county

        # Create semaphore for concurrency control
        semaphore = asyncio.Semaphore(max_concurrent)

        async def update_single_year(year: int) -> Tuple[int, bool, Optional[str]]:
            """Update a single year with error handling."""
            async with semaphore:
                try:
                    print(f"Starting NASS update for {short_desc} {year}...")

                    # Create year-specific parameters
                    year_params = {**base_params, 'year': str(year)}

                    # Debug: Print parameters being passed to API (like api_update)
                    print(f"DEBUG async multi-year params: {year_params}")

                    # Use async query
                    df = await _quickstats_client.query_df_numeric_async(**year_params)

                    if df.empty:
                        error_msg = f"No data returned for {short_desc} {year}"
                        print(f"⚠ Warning: {error_msg}")
                        return year, False, error_msg

                    # Process the data (similar to api_update logic)
                    if 'freq_desc' in df.columns:
                        freqs = df['freq_desc'].unique().tolist()
                        if len(freqs) > 1:
                            if freq_desc_preference in freqs:
                                df = df.loc[df['freq_desc'].str.upper() == freq_desc_preference.upper()]
                            else:
                                # Save the frequency with most datapoints
                                biggest_len = 0
                                largest_fq = None
                                for fq in freqs:
                                    fq_len = len(df.loc[df['freq_desc'] == fq])
                                    biggest_len, largest_fq = (fq_len, fq) if fq_len > biggest_len else (
                                        biggest_len, largest_fq)
                                df = df.loc[df['freq_desc'] == largest_fq]

                    # Date processing (simplified)
                    date_calculation_success = False
                    try:
                        if calc_dates is not None:
                            df_dates = calc_dates(df)
                            df.index = df_dates
                            df['date'] = df.index
                            date_calculation_success = True
                    except Exception as e:
                        print(f'Date calculation failed for {short_desc} {year}: {e}')

                    # Fallback date handling
                    if not date_calculation_success:
                        try:
                            if 'year' in df.columns and 'end_code' in df.columns:
                                df = df[df['end_code'].astype(str).str.isnumeric()]
                                df = df[(df['end_code'].astype(int) >= 1) & (df['end_code'].astype(int) <= 12)]
                                df['date'] = df['year'].astype(str) + '-' + df['end_code'].astype(str).str.zfill(
                                    2) + '-28'
                                df['date'] = pd.to_datetime(df['date'], errors='coerce')
                                df.index = df['date']
                                date_calculation_success = True
                            elif 'year' in df.columns:
                                df['date'] = df['year'].astype(str) + '-12-31'
                                df['date'] = pd.to_datetime(df['date'], errors='coerce')
                                df.index = df['date']
                                date_calculation_success = True
                        except Exception as e:
                            print(f'Fallback date calculation failed for {short_desc} {year}: {e}')

                    # Store the data
                    key = f'{key_desc}/{year}'
                    df.sort_index(inplace=True)
                    df[key_to_name(key_desc)] = clean_col(df['Value'])

                    table_key = key
                    # Prepare columns for storage
                    storage_columns = []
                    if 'date' in df.columns:
                        storage_columns.append('date')
                    if 'commodity_desc' in df.columns:
                        storage_columns.append('commodity_desc')

                    # Add essential state-level columns for non-national data
                    if agg_level != "NATIONAL":
                        if 'state_alpha' in df.columns:
                            storage_columns.append("state_alpha")
                        if 'state_name' in df.columns:
                            storage_columns.append("state_name")
                        if 'state_fips_code' in df.columns:
                            storage_columns.append("state_fips_code")
                        if 'state_ansi' in df.columns:
                            storage_columns.append("state_ansi")

                    # Add essential county-level columns including FIPS codes
                    if agg_level == "COUNTY":
                        if 'county_name' in df.columns:
                            storage_columns.append("county_name")
                        if 'county_code' in df.columns:
                            storage_columns.append("county_code")
                        if 'county_ansi' in df.columns:
                            storage_columns.append("county_ansi")
                        if 'location_desc' in df.columns:
                            storage_columns.append("location_desc")

                    storage_columns.append(key_to_name(key))

                    if 'year' in df.columns:
                        storage_columns.append('year')
                    if 'end_code' in df.columns:
                        storage_columns.append('end_code')

                    # Filter to only existing columns
                    existing_columns = [col for col in storage_columns if col in df.columns]

                    # Store to HDF5
                    df[existing_columns].to_hdf(self.table_db, key=f'{table_key}')
                    print(f"✓ Successfully updated {short_desc} {year} - {len(df)} records")

                    return year, True, None

                except Exception as e:
                    error_msg = f"Failed to update {short_desc} {year}: {str(e)}"
                    print(f"✗ Error: {error_msg}")
                    return year, False, error_msg

        # Execute updates concurrently
        print(f"Updating {short_desc} data for years {start_year}-{end_year} (max {max_concurrent} concurrent)")

        try:
            # Create tasks for all years
            tasks = [update_single_year(year) for year in years]

            # Execute with progress tracking
            completed_results = await asyncio.gather(*tasks, return_exceptions=True)

            # Process results
            for result in completed_results:
                if isinstance(result, Exception):
                    error_year = "unknown"
                    results['failed_years'].append(error_year)
                    results['errors'][error_year] = str(result)
                else:
                    year, success, error_msg = result
                    if success:
                        results['successful_years'].append(year)
                        table_key = f'{self.prefix}/{short_desc.lower()}_{year}' if self.prefix else f'{short_desc.lower()}_{year}'
                        results['updated_tables'].append(table_key)
                    else:
                        results['failed_years'].append(year)
                        if error_msg:
                            results['errors'][year] = error_msg

            # Generate summary
            total_years = len(years)
            successful_count = len(results['successful_years'])
            failed_count = len(results['failed_years'])

            results['summary'] = {
                'short_desc': short_desc,
                'year_range': f"{start_year}-{end_year}",
                'total_years_requested': total_years,
                'successful_updates': successful_count,
                'failed_updates': failed_count,
                'success_rate': f"{successful_count / total_years * 100:.1f}%" if total_years > 0 else "0%",
                'agg_level': agg_level,
                'max_concurrent_requests': max_concurrent
            }

            print(f"\n📊 NASS Update Summary for {short_desc}:")
            print(f"   Years requested: {total_years}")
            print(f"   Successful: {successful_count}")
            print(f"   Failed: {failed_count}")
            print(f"   Success rate: {results['summary']['success_rate']}")

            if results['failed_years']:
                print(f"   Failed years: {results['failed_years']}")

        except Exception as e:
            error_msg = f"Critical error during multi-year update: {str(e)}"
            print(f"💥 {error_msg}")
            results['errors']['critical'] = error_msg

        return results

    def api_update_multi_year_sync(self, short_desc: str, start_year: int, end_year: int, **kwargs) -> Dict[str, any]:
        """
        Synchronous wrapper for the async multi-year update method.
        
        Args:
            short_desc: NASS short description
            start_year: Starting year (inclusive)
            end_year: Ending year (inclusive)
            **kwargs: Additional arguments passed to async method
            
        Returns:
            Dict containing update results
        """
        # Check if we're already in an event loop
        try:
            loop = asyncio.get_running_loop()
            # We're already in an async context, need to create a new task
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor() as executor:
                future = executor.submit(
                    asyncio.run,
                    self.api_update_multi_year_async(short_desc, start_year, end_year, **kwargs)
                )
                return future.result()
        except RuntimeError:
            # No running loop, safe to use asyncio.run
            return asyncio.run(
                self.api_update_multi_year_async(short_desc, start_year, end_year, **kwargs)
            )

    def update_from_url(self, url, key=None, use_prefix=True):
        """
        Update NASSTable from a USDA QuickStats URL.
        
        Args:
            url: USDA QuickStats URL (e.g., https://quickstats.nass.usda.gov/results/...)
            key: Storage key for the data (auto-generated if None)
            use_prefix: Whether to use the table prefix
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            print(f'Fetching data from URL: {url}')

            # Fetch the data from the URL
            import requests
            response = requests.get(url)
            response.raise_for_status()

            # Parse the response - QuickStats URLs typically return CSV data
            from io import StringIO
            df = pd.read_csv(StringIO(response.text))

            if df.empty:
                print(f'No data returned from URL')
                return False

            print(f'Downloaded {len(df)} rows, {len(df.columns)} columns')
            print(f'Columns: {list(df.columns)}')

            # Process the data similar to api_update
            if 'freq_desc' in df.columns:
                freqs = df['freq_desc'].unique().tolist()
                if len(freqs) > 1:
                    # Use the frequency with most datapoints
                    biggest_len = 0
                    largest_fq = None
                    for fq in freqs:
                        fq_len = len(df.loc[df['freq_desc'] == fq])
                        biggest_len, largest_fq = (fq_len, fq) if fq_len > biggest_len else (biggest_len, largest_fq)
                    df = df.loc[df['freq_desc'] == largest_fq]

            # Try to calculate dates
            date_calculation_success = False
            try:
                if calc_dates is not None:
                    df_dates = calc_dates(df)
                    df.index = df_dates
                    df['date'] = df.index
                    date_calculation_success = True
                    if key is None and 'short_desc' in df.columns:
                        key = f'{df["short_desc"].iloc[0].lower()}'
            except Exception as e:
                print(f'Date calculation failed: {e}')

            # Fallback date handling
            if not date_calculation_success:
                try:
                    if 'year' in df.columns and 'end_code' in df.columns:
                        df = df[df['end_code'].astype(str).str.isnumeric()]
                        df = df[(df['end_code'].astype(int) >= 1) & (df['end_code'].astype(int) <= 12)]
                        df['date'] = df['year'].astype(str) + '-' + df['end_code'].astype(str).str.zfill(2) + '-28'
                        df['date'] = pd.to_datetime(df['date'], errors='coerce')
                        df.index = df['date']
                        date_calculation_success = True
                        print(f'Used fallback date calculation')
                    elif 'year' in df.columns:
                        df['date'] = df['year'].astype(str) + '-12-31'
                        df['date'] = pd.to_datetime(df['date'], errors='coerce')
                        df.index = df['date']
                        date_calculation_success = True
                        print(f'Used year-only date calculation')
                except Exception as e:
                    print(f'Fallback date calculation failed: {e}')

            # Generate key if not provided
            if key is None:
                if 'short_desc' in df.columns:
                    key = df['short_desc'].iloc[0].lower().replace(' ', '_').replace(',', '')
                else:
                    key = 'url_data'

            # Store the data
            if not df.empty:
                try:
                    df.sort_index(inplace=True)
                    if 'Value' in df.columns:
                        df[key_to_name(key)] = clean_col(df['Value'])

                    if use_prefix:
                        table_key = f'{self.prefix}/{key}'
                    else:
                        table_key = key

                    # Prepare columns for storage
                    storage_columns = []
                    if 'date' in df.columns:
                        storage_columns.append('date')
                    if 'commodity_desc' in df.columns:
                        storage_columns.append('commodity_desc')
                    if 'short_desc' in df.columns:
                        storage_columns.append('short_desc')
                    if key_to_name(key) in df.columns:
                        storage_columns.append(key_to_name(key))
                    if 'year' in df.columns:
                        storage_columns.append('year')
                    if 'end_code' in df.columns:
                        storage_columns.append('end_code')

                    # Store only existing columns
                    df[storage_columns].to_hdf(self.table_db, key=f'{table_key}')
                    print(f'Data from URL saved to {Path(self.table_db)} in key {table_key}')
                    return True

                except Exception as storage_error:
                    print(f'Error storing data from URL: {storage_error}')
                    return False
            else:
                print(f'No data to store from URL')
                return False

        except Exception as e:
            print(f'Error fetching data from URL: {e}')
            import traceback
            traceback.print_exc()
            return False


class FASTable(TableClient):
    psd = PSD_API()
    comms, attrs, countries = CommodityCode, AttributeCode, CountryCode

    # Shares the same table as NASSTable for convenience’s sake as they link to the same commodities and agency
    def __init__(self, commodity=None):
        super().__init__(client=self.psd, data_folder=DATA_PATH, db_file_name='nass_agri_stats.hd5',
                         key_prefix=commodity, rename_on_load=False)
        self.esr = USDAESR()
        self.prefix = commodity if commodity else None
        self.code = comms_dict[commodity] if commodity else None
        self.country = CountryCode.UNITED_STATES
        self.type = 'livestock' if (self.prefix == 'hog' or self.prefix == 'cattle') else 'grain'
        with open(Path(self.app_path, 'data', 'sources/usda', "esr_map.toml")) as fp:
            data_map = toml.load(fp)
            self.aliases = data_map['aliases']
            self.esr_codes = data_map['esr']
        self.rename = False

    def update_psd(self, commodity=None, start_year=1982, end_year=2025):
        if isinstance(commodity, Enum):
            if rev_lookup.get(commodity, False):
                table_key = rev_lookup[commodity]
                code = commodity.value
            else:
                table_key = commodity.name.lower()
                code = commodity.value
        else:
            if comms_dict.get(commodity, False):
                table_key = commodity
                code = comms_dict.get(commodity)
            elif commodity in valid_codes:
                table_key = code_lookup[commodity].lower()
                code = commodity
            else:
                return print(f'Unable to find valid Commodity/Code for {commodity}')

        years = list(range(start_year, end_year))
        sd_summary = self.client.get_supply_demand_summary(commodity=code, country=self.country, years=years)
        sd_summary.to_hdf(self.table_db, key=f'{table_key}/psd/summary')

        return sd_summary

    # noinspection PyTypeChecker
    def update_table_local(self, data_folder, key_type="imports"):
        aliases = self.aliases
        loading_func = lambda x: clean_fas(x)
        files = os.listdir(data_folder)
        tables_updated = []
        file_aliases = aliases.get(key_type, {})
        for f in files:
            split_str = f.split('_')
            label, split_key = split_str[0], split_str[1]
            split_key = split_key.replace('.csv', '')
            if file_aliases.get(label, False) and split_key == key_type:
                prefix = aliases[key_type][label]
                # noinspection PyTypeChecker
                self.local_update(Path(data_folder, f), new_key=f'{prefix}/{key_type}', use_prefix=False,
                                  load_func=clean_fas)
                tables_updated.append(prefix)
        print(f"Tables Updated: \n{[table for table in tables_updated]}")

        return

    def get_top_sources_by_year(self, commodity, data_type='exports', selected_year=2024):
        if commodity in self.aliases.keys():
            commodity_cat = self.aliases[commodity]
        else:
            commodity_cat = commodity

        df_sources = self[f'{commodity_cat}/{data_type}']
        year_data = df_sources[df_sources.index.get_level_values('date').year == selected_year].copy()
        country_totals = year_data.groupby('Partner')['value'].sum().sort_values(ascending=False)

        return country_totals

    def ESR_top_sources_update(self, commodity, market_year, top_n=10, return_key=True, use_prev_mkt_year=False):
        src_year = market_year - 1 if use_prev_mkt_year or market_year == datetime.datetime.today().year else market_year
        top_res = self.get_top_sources_by_year(commodity, selected_year=market_year)
        top_destinations = top_res.index.to_list()

        # Translate commodity table name to ESR commodity code
        # Step 1: commodity (e.g., "cattle") -> commodity_name (e.g., "beef") via esr.alias 
        # Step 2: commodity_name (e.g., "beef") -> commodity_code (e.g., "1701") via esr.commodities

        if commodity.lower() in self.esr_codes['alias'].keys():
            # Get the commodity name from the alias (cattle -> beef, hogs -> pork, etc.)
            commodity_name = self.esr_codes['alias'][commodity.lower()]
            # Get the commodity code from the commodity name (beef -> 1701, pork -> 1702, etc.)
            commodity_code = self.esr_codes['commodities'][commodity_name]
            table_name = commodity.lower()
        else:
            # Fallback: assume commodity is already the commodity name (beef, pork, etc.)
            commodity_code = self.esr_codes['commodities'].get(commodity.lower())
            if not commodity_code:
                raise ValueError(f"Unknown commodity: {commodity}. Available: {list(self.esr_codes['alias'].keys())}")
            table_name = commodity.lower()

        dest_dfs = []
        failed_countries = []

        for t in top_destinations[:top_n]:
            t = t[:-3] if t.endswith('(*)') else t[:]
            if self.esr_codes['countries'].get(t.upper(), False):
                country_code = self.esr_codes['countries'][t.upper()]
                country_df = self.esr.get_export_sales(commodity_code=commodity_code, country_code=country_code,
                                                       market_year=market_year)
                try:
                    country_df.index = pd.to_datetime(country_df['weekEndingDate'])
                except KeyError:
                    print(f"Failed to get country {t}")
                    if country_df.empty:
                        print('Failed to get data from server')
                    failed_countries.append(t)
                else:
                    country_df['country'] = t
                    country_df['commodity'] = commodity
                    dest_dfs.append(country_df)

            else:
                print(f'Failed to get country {t}')
                failed_countries.append(t)
        if dest_dfs:
            export_year_df = pd.concat(dest_dfs, axis=0)
            export_year_df.to_hdf(self.table_db, f'{table_name}/exports/{market_year}')
            return_data = export_year_df if return_key else None
            return return_data

        else:
            print(f'Failed to get data for year {market_year} ')


class WeatherTable(TableClient):

    def __init__(self, client, data_folder, db_file_name, key_prefix=None, map_file=None):
        return


# Used for tracking finished good exports (Beef, Pork) and Unfinished goods (Corn, Soy) exports
class ESRTableClient(FASTable):
    """
    Extended TableClient specifically for ESR data handling.
    Includes methods for ESR-specific data processing and filtering.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def get_esr_data(self, commodity, year=None, country=None,
                     start_date=None, end_date=None):
        """
        Get ESR data with optional filtering.

        Args:
            commodity: Commodity name (e.g., 'wheat', 'corn', 'soybeans')
            year: Marketing year (optional)
            country: Country filter (optional)
            start_date: Start date filter (optional)
            end_date: End date filter (optional)

        Returns:
            pd.DataFrame: Filtered ESR data
        """
        # Construct key based on ESR format: {commodity}/exports/{year}
        if year:
            key = f"{commodity}/exports/{year}"
        else:
            # Get most recent year available
            available_keys = self.available_keys()
            commodity_keys = [k for k in available_keys if k.startswith(f"/{commodity}/exports/")]
            if not commodity_keys:
                return pd.DataFrame()

            # Get latest year
            years = [int(k.split('/')[-1]) for k in commodity_keys if k.split('/')[-1].isdigit()]
            if not years:
                return pd.DataFrame()

            latest_year = max(years)
            key = f"{commodity}/exports/{latest_year}"

        # Get base data
        data = self.get_key(key)

        if data is None or data.empty:
            return pd.DataFrame()

        # Apply filters
        if country:
            data = data[data['country'].str.contains(country, case=False, na=False)]

        if start_date:
            data = data[pd.to_datetime(data['weekEndingDate']) >= pd.to_datetime(start_date)]

        if end_date:
            data = data[pd.to_datetime(data['weekEndingDate']) <= pd.to_datetime(end_date)]

        return data

    def get_multi_year_esr_data(self, commodity, years=None, country=None,
                                start_year=None, end_year=None):
        """
        Get ESR data concatenated from multiple years.

        Args:
            commodity: Commodity name
            years: List of years (if None, uses start_year/end_year or last 5 years)
            country: Optional country filter
            start_year: Start year for range (alternative to years list)
            end_year: End year for range (alternative to years list)

        Returns:
            pd.DataFrame: Concatenated ESR data
        """
        if years is None:
            if start_year and end_year:
                years = list(range(start_year, end_year + 1))
            else:
                current_year = pd.Timestamp.now().year
                years = list(range(current_year - 4, current_year + 1))

        all_data = []

        for year in years:
            year_data = self.get_esr_data(commodity, year, country)
            if not year_data.empty:
                # Standardize columns_col names - some years have different columns_col structures
                year_data = self._standardize_esr_columns(year_data, year)
                year_data['marketing_year'] = year
                all_data.append(year_data)

        if all_data:
            combined_data = pd.concat(all_data, ignore_index=True)
            # Only sort by weekEndingDate if it exists
            if 'weekEndingDate' in combined_data.columns:
                combined_data = combined_data.sort_values(['marketing_year', 'weekEndingDate'])
            else:
                combined_data = combined_data.sort_values(['marketing_year'])
            return combined_data

        return pd.DataFrame()

    def _standardize_esr_columns(self, data: pd.DataFrame, year: int) -> pd.DataFrame:
        """
        Standardize ESR columns_col names and fix unit scaling issues across different years.
        
        Some years may have different columns_col naming conventions (e.g., '2025_exports'
        instead of 'weekEndingDate') and different unit scaling (grains may have 1000x scaling differences).
        
        Args:
            data: DataFrame to standardize
            year: Year of the data for context
            
        Returns:
            DataFrame with standardized columns_col names and units
        """
        data = data.copy()

        # Check for year-specific date columns (like '2025_exports', '2024_exports')
        year_column = f"{year}_exports"
        if year_column in data.columns and 'weekEndingDate' not in data.columns:
            # Rename the year columns_col to weekEndingDate
            data = data.rename(columns={year_column: 'weekEndingDate'})
            print(f"Standardized columns_col '{year_column}' to 'weekEndingDate' for {year} data")

        # Ensure weekEndingDate is properly formatted as datetime
        if 'weekEndingDate' in data.columns:
            try:
                data['weekEndingDate'] = pd.to_datetime(data['weekEndingDate'])
            except Exception as e:
                print(f"Warning: Could not convert weekEndingDate to datetime for {year}: {e}")

        # Fix unit scaling issues for grains
        if 'commodity' in data.columns:
            commodity = data['commodity'].iloc[0].lower() if not data.empty else ''
            # Grains (corn, wheat, soybeans) sometimes have 1000x scaling differences between years
            grain_commodities = ['corn', 'wheat', 'soybeans']

            if any(grain in commodity for grain in grain_commodities):
                # Check if this year's data appears to be in wrong units (too large values)
                numeric_columns = ['weeklyExports', 'outstandingSales', 'grossNewSales',
                                   'currentMYNetSales', 'currentMYTotalCommitment',
                                   'nextMYOutstandingSales', 'nextMYNetSales', 'accumulatedExports']

                for col in numeric_columns:
                    if col in data.columns:
                        # Check if values are abnormally high (indicating wrong units)
                        mean_value = data[col].mean()
                        max_value = data[col].max()

                        # If mean > 1000 or max > 100000, likely needs scaling down by 1000
                        if mean_value > 1000 or max_value > 100000:
                            data[col] = data[col] / 1000.0
                            print(f"Applied 1000x scaling correction to {col} for {commodity} {year} data")

        # Remove any duplicate date columns that might exist
        date_like_columns = [col for col in data.columns if 'exports' in col and col != 'weekEndingDate']
        for col in date_like_columns:
            if col in data.columns and 'weekEndingDate' in data.columns:
                # If we already have weekEndingDate, drop the redundant columns_col
                data = data.drop(columns=[col])
                print(f"Removed redundant date columns_col '{col}' from {year} data")

        return data

    def get_available_commodities(self):
        """Get list of available commodities in ESR data."""
        available_keys = self.available_keys()
        commodities = set()

        for key in available_keys:
            if '/exports/' in key:
                parts = key.split('/')
                if len(parts) >= 2:
                    commodity = parts[1]  # Remove leading /
                    commodities.add(commodity)

        return sorted(list(commodities))

    def get_available_years(self, commodity):
        """Get available years for a specific commodity."""
        available_keys = self.available_keys()
        years = set()

        for key in available_keys:
            if f"{commodity}/exports/" in key:
                parts = key.split('/')
                if len(parts) >= 3 and parts[-1].isdigit():
                    years.add(int(parts[-1]))

        return sorted(list(years))

    def get_available_countries(self, commodity, year=None):
        """Get available countries for a commodity."""
        data = self.get_esr_data(commodity, year)
        if not data.empty and 'country' in data.columns:
            return sorted(data['country'].unique().tolist())
        return []

    def get_top_countries(self, commodity, metric='weeklyExports', top_n=10,
                          start_year=None, end_year=None):
        """
        Get top N countries by export metric for dynamic menu population.
        
        Args:
            commodity: Commodity name
            metric: Metric to rank by ('weeklyExports', 'outstandingSales', etc.)
            top_n: Number of top countries to return
            start_year: Start year for analysis (optional)
            end_year: End year for analysis (optional)
            
        Returns:
            List of country names sorted by metric (descending)
        """
        try:
            # Get multi-year data for ranking
            data = self.get_multi_year_esr_data(commodity, start_year=start_year, end_year=end_year)

            if data.empty or metric not in data.columns:
                return []

            # Aggregate by country
            country_totals = data.groupby('country')[metric].sum().sort_values(ascending=False)

            # Return top N countries
            top_countries = country_totals.head(top_n).index.tolist()

            return top_countries

        except Exception as e:
            print(f"Error getting top countries: {e}")
            return []

    def get_commitment_vs_shipment_analysis(self, commodity, start_year=None, end_year=None, countries=None):
        """
        Get commitment vs shipment analysis using ESRAnalyzer.
        
        Args:
            commodity: Commodity name
            start_year: Start year for analysis
            end_year: End year for analysis 
            countries: List of countries to include (optional)
            
        Returns:
            Dict with analysis results including 'data' key with processed DataFrame
        """
        try:
            from MacrOSINT.models.agricultural.agricultural_analytics import ESRAnalyzer

            # Get multi-year data
            data = self.get_multi_year_esr_data(commodity, start_year=start_year, end_year=end_year)

            if data.empty:
                return {'error': 'no_data'}

            # Filter by countries if specified
            if countries:
                data = data[data['country'].isin(countries)]

            # Determine commodity type for analyzer
            grain_commodities = ['corn', 'wheat', 'soybeans']
            oilseed_commodities = ['soybeans']  # soybeans can be both
            livestock_commodities = ['cattle', 'hogs', 'pork']

            if commodity.lower() in grain_commodities:
                commodity_type = 'grains'
            elif commodity.lower() in livestock_commodities:
                commodity_type = 'livestock'
            else:
                commodity_type = 'grains'  # default

            # Initialize ESR analyzer
            analyzer = ESRAnalyzer(data.set_index('weekEndingDate'), commodity_type)

            # Run commitment vs shipment analysis
            results = analyzer.commitment_vs_shipment_analysis()

            return results

        except Exception as e:
            print(f"Error in commitment vs shipment analysis: {e}")
            return {'error': str(e)}

    def get_seasonal_patterns_analysis(self, commodity, metric='weeklyExports', start_year=None, end_year=None,
                                       countries=None):
        """
        Get seasonal patterns analysis using ESRAnalyzer.
        
        Args:
            commodity: Commodity name
            metric: ESR metric to analyze for seasonality
            start_year: Start year for analysis
            end_year: End year for analysis
            countries: List of countries to include (optional)
            
        Returns:
            Dict with seasonal analysis results including processed data
        """
        try:
            from MacrOSINT.models.agricultural.agricultural_analytics import ESRAnalyzer

            # Get multi-year data
            data = self.get_multi_year_esr_data(commodity, start_year=start_year, end_year=end_year)

            if data.empty:
                return {'error': 'no_data'}

            # Filter by countries if specified
            if countries:
                data = data[data['country'].isin(countries)]

            # Determine commodity type
            grain_commodities = ['corn', 'wheat', 'soybeans']
            livestock_commodities = ['cattle', 'hogs', 'pork']

            if commodity.lower() in grain_commodities:
                commodity_type = 'grains'
            elif commodity.lower() in livestock_commodities:
                commodity_type = 'livestock'
            else:
                commodity_type = 'grains'

            # Initialize ESR analyzer
            analyzer = ESRAnalyzer(data.set_index('weekEndingDate'), commodity_type)

            # Run seasonal analysis
            seasonal_results = analyzer.analyze_seasonal_patterns(metric)

            # Add the original data with marketing year week for plotting
            analysis_data = analyzer.data.copy()
            seasonal_results['data'] = analysis_data.reset_index()

            return seasonal_results

        except Exception as e:
            print(f"Error in seasonal patterns analysis: {e}")
            return {'error': str(e)}

    def aggregate_esr_data(self, data, group_by='country', time_period='weekly'):
        """
        Aggregate ESR data by different dimensions.

        Args:
            data: ESR DataFrame
            group_by: 'country', 'commodity', or 'month'
            time_period: 'weekly', 'monthly', 'quarterly'

        Returns:
            pd.DataFrame: Aggregated data
        """
        if data.empty:
            return pd.DataFrame()

        # Ensure weekEndingDate is datetime
        data['weekEndingDate'] = pd.to_datetime(data['weekEndingDate'])

        # Create time grouping columns_col
        if time_period == 'monthly':
            data['time_group'] = data['weekEndingDate'].dt.to_period('M')
        elif time_period == 'quarterly':
            data['time_group'] = data['weekEndingDate'].dt.to_period('Q')
        else:  # weekly
            data['time_group'] = data['weekEndingDate']

        # Define aggregation functions
        agg_functions = {
            'weeklyExports': 'sum',
            'accumulatedExports': 'last',
            'outstandingSales': 'last',
            'grossNewSales': 'sum',
            'currentMYNetSales': 'sum',
            'currentMYTotalCommitment': 'last',
            'nextMYOutstandingSales': 'last',
            'nextMYNetSales': 'sum'
        }

        # Group and aggregate
        if group_by == 'country':
            grouped = data.groupby(['time_group', 'country']).agg(agg_functions).reset_index()
        elif group_by == 'commodity':
            grouped = data.groupby(['time_group', 'commodity']).agg(agg_functions).reset_index()
        else:  # total
            grouped = data.groupby('time_group').agg(agg_functions).reset_index()

        return grouped

    async def ESR_multi_year_update(self, commodity: str, start_year: int, end_year: int,
                                    top_n: int = 10, max_concurrent: int = 3) -> Dict[str, any]:
        """
        Update ESR data for multiple years using async/await for parallelization.
        Uses asyncio instead of threading because:
        - I/O-bound operations (API calls) benefit more from async
        - Better resource efficiency (single thread + event loop)
        - Natural error propagation and handling
        - Built-in concurrency control
        
        Args:
            commodity: Commodity name
            start_year: Starting marketing year
            end_year: Ending marketing year  
            top_n: Number of top countries to update (default: 10)
            max_concurrent: Maximum concurrent requests (default: 3)
            
        Returns:
            Dict containing success/failure results and summary statistics
        """
        years = list(range(start_year, end_year + 1))
        results = {
            'successful_years': [],
            'failed_years': [],
            'errors': {},
            'updated_tables': [],
            'summary': {}
        }

        # Create semaphore to limit concurrent requests
        semaphore = asyncio.Semaphore(max_concurrent)

        async def update_single_year(year: int) -> Tuple[int, bool, Optional[str]]:
            """Update a single year with error handling."""
            async with semaphore:  # Limit concurrent requests
                try:
                    print(f"Starting update for {commodity} {year}...")

                    # Use existing synchronous method but wrap in executor if needed
                    # For now, we'll call the sync method in a thread pool
                    loop = asyncio.get_event_loop()

                    # Call the existing ESR_top_sources_update method
                    result = await loop.run_in_executor(
                        None,
                        lambda: self.ESR_top_sources_update(
                            commodity=commodity,
                            market_year=year,
                            top_n=top_n,
                            return_key=True
                        )
                    )

                    if result is not None and not result.empty:
                        print(f"✓ Successfully updated {commodity} {year} - {len(result)} records")
                        return year, True, None
                    else:
                        error_msg = f"No data returned for {commodity} {year}"
                        print(f"⚠ Warning: {error_msg}")
                        return year, False, error_msg

                except Exception as e:
                    error_msg = f"Failed to update {commodity} {year}: {str(e)}"
                    print(f"✗ Error: {error_msg}")
                    return year, False, error_msg

        # Execute updates concurrently
        print(f"Updating {commodity} data for years {start_year}-{end_year} (max {max_concurrent} concurrent)")

        try:
            # Create tasks for all years
            tasks = [update_single_year(year) for year in years]

            # Execute with progress tracking
            completed_results = await asyncio.gather(*tasks, return_exceptions=True)

            # Process results
            for result in completed_results:
                if isinstance(result, Exception):
                    error_year = "unknown"
                    results['failed_years'].append(error_year)
                    results['errors'][error_year] = str(result)
                else:
                    year, success, error_msg = result
                    if success:
                        results['successful_years'].append(year)
                        results['updated_tables'].append(f"{commodity}/exports/{year}")
                    else:
                        results['failed_years'].append(year)
                        if error_msg:
                            results['errors'][year] = error_msg

            # Generate summary
            total_years = len(years)
            successful_count = len(results['successful_years'])
            failed_count = len(results['failed_years'])

            results['summary'] = {
                'commodity': commodity,
                'year_range': f"{start_year}-{end_year}",
                'total_years_requested': total_years,
                'successful_updates': successful_count,
                'failed_updates': failed_count,
                'success_rate': f"{successful_count / total_years * 100:.1f}%" if total_years > 0 else "0%",
                'top_n_countries': top_n,
                'max_concurrent_requests': max_concurrent
            }

            print(f"\n📊 Update Summary for {commodity}:")
            print(f"   Years requested: {total_years}")
            print(f"   Successful: {successful_count}")
            print(f"   Failed: {failed_count}")
            print(f"   Success rate: {results['summary']['success_rate']}")

            if results['failed_years']:
                print(f"   Failed years: {results['failed_years']}")

        except Exception as e:
            error_msg = f"Critical error during multi-year update: {str(e)}"
            print(f"💥 {error_msg}")
            results['errors']['critical'] = error_msg

        return results

    def update_esr_multi_year_sync(self, commodity: str, start_year: int, end_year: int,
                                   top_n: int = 10, max_concurrent: int = 3) -> Dict[str, any]:
        """
        Synchronous wrapper for the async multi-year update method.
        
        Args:
            commodity: Commodity name
            start_year: Starting marketing year
            end_year: Ending marketing year
            top_n: Number of top countries to update (default: 10)
            max_concurrent: Maximum concurrent requests (default: 3)
            
        Returns:
            Dict containing update results
        """
        # Check if we're already in an event loop
        try:
            loop = asyncio.get_running_loop()
            # We're already in an async context, need to create a new task
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor() as executor:
                future = executor.submit(
                    asyncio.run,
                    self.ESR_multi_year_update(commodity, start_year, end_year, top_n, max_concurrent)
                )
                return future.result()
        except RuntimeError:
            # No running loop, safe to use asyncio.run
            return asyncio.run(
                self.ESR_multi_year_update(commodity, start_year, end_year, top_n, max_concurrent)
            )

    def update_esr(self, data: pd.DataFrame, commodity: str, year: int):
        if commodity.lower() in self.esr_codes['alias'].keys():
            # Get the commodity name from the alias (cattle -> beef, hogs -> pork, etc.)
            commodity_name = self.esr_codes['alias'][commodity.lower()]
            # Get the commodity code from the commodity name (beef -> 1701, pork -> 1702, etc.)
            commodity_code = self.esr_codes['commodities'][commodity_name]
            table_name = commodity.lower()
        else:
            # Fallback: assume commodity is already the commodity name (beef, pork, etc.)
            commodity_code = self.esr_codes['commodities'].get(commodity.lower())
            if not commodity_code:
                raise ValueError(f"Unknown commodity: {commodity}. Available: {list(self.esr_codes['alias'].keys())}")

            table_name = commodity.lower()
        if not data.empty:
            data.to_hdf(self.table_db, key=f"{commodity}/exports/{year}")
            print(f'Data Successfully updated to table {commodity}/exports/{year}')
            return True
        else:
            return False

    def update_esr_from_csv(self, csv_file_path: str, target_year: int = None,
                            validate: bool = True) -> Dict[str, any]:
        """
        Update ESR export data from a downloaded CSV file that may contain multiple commodities.
        
        This method consumption_processes a raw ESR CSV download, detects all commodities present,
        cleans the data to standard format, and updates the appropriate 
        {commodity}/exports/{year} tables for each commodity found.
        
        Args:
            csv_file_path (str): Path to the raw ESR CSV file
            target_year (int, optional): Marketing year (auto-detected if None)  
            validate (bool): Whether to validate data before updating (default True)
            
        Returns:
            Dict containing update results for all commodities processed
            
        Example:
            result = client.update_esr_from_csv("esr_mixed_2025.csv")
            for commodity, commodity_result in result['commodities'].items():
                print(f"Updated {commodity}: {commodity_result['records_count']} records")
        """
        if not process_esr_csv:
            raise ImportError("ESR CSV processor not available. Check imports.")

        try:
            print(f"🔄 Processing multi-commodity ESR CSV: {csv_file_path}")

            # Read the raw CSV to detect commodities
            raw_df = pd.read_csv(csv_file_path)
            print(f"📊 Raw data shape: {raw_df.shape}")

            # Detect all commodities in the file
            detected_commodities = self._detect_all_commodities(raw_df)
            print(f"🎯 Detected commodities: {detected_commodities}")

            if not detected_commodities:
                raise ValueError("No recognizable commodities found in CSV file")

            # Process each commodity separately
            commodity_results = {}
            overall_success = True
            total_records = 0

            for commodity in detected_commodities:
                print(f"\n📦 Processing commodity: {commodity}")

                try:
                    # Filter data for this specific commodity
                    commodity_data = self._filter_csv_by_commodity(raw_df, commodity)

                    if commodity_data.empty:
                        print(f"⚠️  No data found for {commodity}")
                        continue

                    # Process and clean the commodity-specific data
                    processor = ESRCSVProcessor()
                    cleaned_data = processor._transform_raw_data(commodity_data, commodity)

                    # Extract year from the data
                    detected_year = processor.extract_year_from_data(cleaned_data)
                    final_year = target_year if target_year is not None else detected_year

                    print(f"📈 {commodity}: {len(cleaned_data)} records for year {final_year}")

                    # Validate data if requested
                    validation_issues = []
                    if validate:
                        is_valid, issues = processor.validate_data(cleaned_data)
                        validation_issues = issues

                        if not is_valid:
                            print(f"⚠️  Validation warnings for {commodity}:")
                            for issue in issues:
                                print(f"    - {issue}")

                    # Update the data table for this commodity
                    update_success = self.update_esr(cleaned_data, commodity, final_year)

                    # Store results for this commodity
                    commodity_results[commodity] = {
                        'success': update_success,
                        'commodity': commodity,
                        'year': final_year,
                        'records_count': len(cleaned_data),
                        'countries_count': cleaned_data['country'].nunique() if not cleaned_data.empty else 0,
                        'date_range': {
                            'start': cleaned_data['weekEndingDate'].min().strftime(
                                '%Y-%m-%d') if not cleaned_data.empty else None,
                            'end': cleaned_data['weekEndingDate'].max().strftime(
                                '%Y-%m-%d') if not cleaned_data.empty else None
                        },
                        'validation_issues': validation_issues,
                        'table_key': f"{commodity}/exports/{final_year}"
                    }

                    if update_success:
                        print(f"✅ Successfully updated {commodity} data for {final_year}")
                        total_records += len(cleaned_data)
                    else:
                        print(f"❌ Failed to update {commodity} data for {final_year}")
                        overall_success = False

                except Exception as e:
                    print(f"❌ Error processing {commodity}: {e}")
                    commodity_results[commodity] = {
                        'success': False,
                        'error': str(e),
                        'commodity': commodity
                    }
                    overall_success = False

            # Prepare overall results
            successful_commodities = [c for c, r in commodity_results.items() if r.get('success', False)]
            failed_commodities = [c for c, r in commodity_results.items() if not r.get('success', False)]

            result = {
                'overall_success': overall_success,
                'total_commodities': len(detected_commodities),
                'successful_commodities': successful_commodities,
                'failed_commodities': failed_commodities,
                'total_records_processed': total_records,
                'commodities': commodity_results,
                'file_processed': csv_file_path
            }

            print(f"\n🎯 Multi-commodity update completed:")
            print(f"   - Total commodities: {len(detected_commodities)}")
            print(f"   - Successful: {len(successful_commodities)}")
            print(f"   - Failed: {len(failed_commodities)}")
            print(f"   - Total records: {total_records}")

            return result

        except Exception as e:
            error_result = {
                'overall_success': False,
                'error': str(e),
                'total_commodities': 0,
                'commodities': {},
                'file_processed': csv_file_path
            }
            print(f"❌ Error updating ESR from CSV: {e}")
            return error_result

    def _detect_all_commodities(self, raw_df: pd.DataFrame) -> List[str]:
        """
        Detect all commodities present in the raw CSV data.
        
        Args:
            raw_df: Raw DataFrame from CSV
            
        Returns:
            List of commodity names found in the data
        """
        commodities = set()

        if 'Commodity' in raw_df.columns:
            # Map raw commodity names to our internal names
            commodity_mapping = {
                'ALL WHEAT': 'wheat',
                'WHEAT': 'wheat',
                'CORN': 'corn',
                'CORN - UNMILLED': 'corn',
                'SOYBEANS': 'soybeans',
                'FRESH, CHILLED, OR FROZEN MUSCLE CUTS OF BEEF': 'cattle',
                'CATTLE': 'cattle',
                'FRESH, CHILLED, OR FROZEN MUSCLE CUTS OF PORK': 'hogs',
                'HOGS': 'hogs',
                'RICE': 'rice',
                'SORGHUM': 'sorghum',
                'BARLEY': 'barley'
            }

            raw_commodities = raw_df['Commodity'].unique()

            for raw_commodity in raw_commodities:
                if raw_commodity in commodity_mapping:
                    commodities.add(commodity_mapping[raw_commodity])
                else:
                    # Try to clean and add unknown commodities
                    cleaned = str(raw_commodity).lower().strip()
                    if cleaned and len(cleaned) > 1:
                        commodities.add(cleaned)

        return sorted(list(commodities))

    def _filter_csv_by_commodity(self, raw_df: pd.DataFrame, target_commodity: str) -> pd.DataFrame:
        """
        Filter raw CSV data for a specific commodity.
        
        Args:
            raw_df: Raw DataFrame from CSV
            target_commodity: Target commodity to filter for
            
        Returns:
            DataFrame filtered for the specific commodity
        """
        if 'Commodity' not in raw_df.columns:
            return raw_df

        # Reverse mapping to find raw commodity names
        commodity_mapping = {
            'wheat': ['ALL WHEAT', 'WHEAT'],
            'corn': ['CORN', 'CORN - UNMILLED'],
            'soybeans': ['SOYBEANS'],
            'cattle': ['FRESH, CHILLED, OR FROZEN MUSCLE CUTS OF BEEF', 'CATTLE'],
            'hogs': ['FRESH, CHILLED, OR FROZEN MUSCLE CUTS OF PORK', 'HOGS'],
            'rice': ['RICE'],
            'sorghum': ['SORGHUM'],
            'barley': ['BARLEY']
        }

        # Find raw commodity names that match our target
        target_raw_names = commodity_mapping.get(target_commodity, [target_commodity.upper()])

        # Filter for matching commodities
        mask = raw_df['Commodity'].isin(target_raw_names)
        filtered_df = raw_df[mask].copy()

        return filtered_df

    def merge_export_years(self, commodity, save_merged=True):
        """
        Merge all export years for a commodity into a single /exports/all key.
        
        Args:
            commodity: Commodity name (e.g., 'wheat', 'corn', 'soybeans')
            save_merged: Whether to save the merged data to /exports/all key
            
        Returns:
            pd.DataFrame: Merged data from all available export years
        """
        print(f"Merging export years for {commodity}...")

        # Get available years for this commodity
        available_years = self.get_available_years(commodity)

        if not available_years:
            print(f"No export years found for {commodity}")
            return pd.DataFrame()

        print(f"Found years: {available_years}")

        all_data = []
        successful_years = []

        for year in available_years:
            try:
                # Get data for this year
                year_data = self.get_key(f"{commodity}/exports/{year}")

                if year_data is not None and not year_data.empty:
                    # Standardize the data
                    year_data = self._standardize_esr_columns(year_data, year)
                    year_data['marketing_year'] = year
                    all_data.append(year_data)
                    successful_years.append(year)
                    print(f"  [OK] {year}: {len(year_data)} rows")
                else:
                    print(f"  [WARNING] {year}: No data")

            except Exception as e:
                print(f"  [ERROR] {year}: {e}")
                continue

        if not all_data:
            print(f"No data could be loaded for {commodity}")
            return pd.DataFrame()

        # Combine all years
        merged_data = pd.concat(all_data, ignore_index=True)

        # Sort by marketing year and date
        if 'weekEndingDate' in merged_data.columns:
            merged_data = merged_data.sort_values(['marketing_year', 'weekEndingDate'])
        else:
            merged_data = merged_data.sort_values(['marketing_year'])

        print(f"Merged data: {len(merged_data)} total rows from {len(successful_years)} years")

        # Save to /exports/all key if requested
        if save_merged:
            try:
                all_key = f"{commodity}/exports/all"
                with pd.HDFStore(self.table_db, mode='a') as store:
                    store.put(f'/{all_key}', merged_data, format='table', data_columns=True)
                print(f"[OK] Saved merged data to /{all_key}")
            except Exception as e:
                print(f"[ERROR] Failed to save merged data: {e}")

        return merged_data

    def get_merged_export_data(self, commodity, force_refresh=False):
        """
        Get merged export data for a commodity, creating it if it doesn't exist.
        
        Args:
            commodity: Commodity name
            force_refresh: Whether to regenerate the merged data even if it exists
            
        Returns:
            pd.DataFrame: Merged export data from all years
        """
        all_key = f"{commodity}/exports/all"

        # Check if merged data already exists
        if not force_refresh:
            try:
                existing_data = self.get_key(all_key)
                if existing_data is not None and not existing_data.empty:
                    print(f"Using existing merged data for {commodity}: {len(existing_data)} rows")
                    return existing_data
            except:
                pass

        # Generate merged data
        return self.merge_export_years(commodity, save_merged=True)


# =============================================================================
# COMMODITY-SPECIFIC EIA SUBCLASSES  
# =============================================================================

class PETable(EIATable):
    """
    Petroleum Liquids specific EIA table client.
    Handles petroleum stocks, production, consumption, and movements.
    """
    
    def __init__(self, rename_key_cols=True):
        super().__init__("PET", rename_key_cols)
    
    # Copy petroleum-specific methods from EIATable
    def update_petroleum_stocks(self, start="2001-01", end="2025-08", max_concurrent=3):
        """Update petroleum stocks using async methods for improved performance."""
        return super().update_petroleum_stocks(start, end, max_concurrent)
    
    async def update_petroleum_stocks_async(self, start="2001-01", end="2025-08", max_concurrent=3):
        """Async version of update_petroleum_stocks for improved performance."""
        return await super().update_petroleum_stocks_async(start, end, max_concurrent)
    
    def update_petroleum_production(self, start=None, end=None):
        """Update petroleum production data."""
        return super().update_petroleum_production(start, end)
    
    def update_petroleum_consumption(self, start=None, end=None):
        """Update petroleum consumption data."""
        return super().update_petroleum_consumption(start, end)
    
    def update_petroleum_movements(self, start=None, end=None):
        """Update petroleum movements data."""
        return super().update_petroleum_movements(start, end)
    
    async def update_petroleum_movements_async(self, start=None, end=None, max_concurrent=3, progress_callback=None):
        """Async version of update_petroleum_movements."""
        return await super().update_petroleum_movements_async(start, end, max_concurrent, progress_callback)
    
    def update_petroleum_movements_async_sync(self, start=None, end=None, max_concurrent=3, progress_callback=None):
        """Synchronous wrapper for async petroleum movements update."""
        return super().update_petroleum_movements_async_sync(start, end, max_concurrent, progress_callback)
    
    def update_weekly_trade(self, start=None, end=None):
        """Function to update import/export tables"""
        return super().update_weekly_trade(start, end)
    
    def update_monthly_trade(self, start=None, end=None):
        """Function to update import/export tables for district data"""
        return super().update_monthly_trade(start, end)
    
    async def update_monthly_trade_async(self, start=None, end=None, max_concurrent=3, progress_callback=None):
        """Async version of update_monthly_trade."""
        return await super().update_monthly_trade_async(start, end, max_concurrent, progress_callback)
    
    def update_monthly_trade_async_sync(self, start=None, end=None, max_concurrent=3, progress_callback=None):
        """Synchronous wrapper for async monthly trade update."""
        return super().update_monthly_trade_async_sync(start, end, max_concurrent, progress_callback)
    
    def update_refinery_data(self, start=None, end=None):
        """Update refinery data."""
        return super().update_refinery_data(start, end)


class NGTable(EIATable):
    """
    Natural Gas specific EIA table client.
    Handles natural gas storage, consumption, production, and trade.
    """
    
    def __init__(self, rename_key_cols=True):
        super().__init__("NG", rename_key_cols)
    
    # Copy natural gas-specific methods from EIATable
    def update_consumption_by_state(self, save=True, key="consumption/by_state", prefix=True, table=None):
        """Update natural gas consumption by state."""
        return super().update_consumption_by_state(save, key, prefix, table)
    
    def get_consumption_by_region(self, start="2001-01-31", end="2025-05-31", end_use=None, update_table=False):
        """Get natural gas consumption data by region and end use."""
        return super().get_consumption_by_region(start, end, end_use, update_table)
    
    def update_natural_gas_storage(self, start=None, end=None):
        """
        Update natural gas storage data.
        
        Args:
            start: Start date in YYYY-MM format
            end: End date in YYYY-MM format
        """
        if not start:
            start = "2001-01"
        if not end:
            end = "2025-08"
        
        helper = self.clients["NG"]
        
        # Get storage data
        try:
            storage_data = helper.get_storage_data(start, end)
            storage_data.to_hdf(self.table_db, "NG/storage/underground")
            print(f"Updated NG storage data: {len(storage_data)} records")
        except Exception as e:
            print(f"Failed to update NG storage data: {e}")
        
        return
    
    def update_natural_gas_production(self, start=None, end=None):
        """
        Update natural gas production data.
        
        Args:
            start: Start date in YYYY-MM format
            end: End date in YYYY-MM format
        """
        if not start:
            start = "2001-01"
        if not end:
            end = "2025-08"
        
        helper = self.clients["NG"]
        
        # Get production data
        try:
            production_data = helper.get_production_data(start, end)
            production_data.to_hdf(self.table_db, "NG/production/dry_gas")
            print(f"Updated NG production data: {len(production_data)} records")
        except Exception as e:
            print(f"Failed to update NG production data: {e}")
        
        return
    
    def update_natural_gas_trade(self, start=None, end=None):
        """
        Update natural gas import/export trade data.
        
        Args:
            start: Start date in YYYY-MM format
            end: End date in YYYY-MM format
        """
        if not start:
            start = "2001-01"
        if not end:
            end = "2025-08"
        
        helper = self.clients["NG"]
        
        # Get trade data
        try:
            imports_data = helper.get_imports_data(start, end)
            exports_data = helper.get_exports_data(start, end)
            
            imports_data.to_hdf(self.table_db, "NG/trade/imports")
            exports_data.to_hdf(self.table_db, "NG/trade/exports")
            
            print(f"Updated NG imports data: {len(imports_data)} records")
            print(f"Updated NG exports data: {len(exports_data)} records")
        except Exception as e:
            print(f"Failed to update NG trade data: {e}")
        
        return
    
    # Async update methods using NatGasHelper
    async def update_underground_storage_async(self, start=None, end=None, max_concurrent=3):
        """
        Update underground natural gas storage data asynchronously.
        
        Args:
            start: Start date in YYYY-MM format
            end: End date in YYYY-MM format
            max_concurrent: Maximum concurrent requests
            
        Returns:
            Dictionary with update results
        """
        if not start:
            start = "2001-01"
        if not end:
            end = "2025-08"
        
        helper = self.clients["NG"]
        
        try:
            storage_data = await helper.get_underground_storage_async(start, end, max_concurrent)
            storage_data.to_hdf(self.table_db, "NG/storage/underground")
            
            result = {
                'success': True,
                'records': len(storage_data),
                'key': 'NG/storage/underground'
            }
            print(f"✓ Updated NG underground storage: {len(storage_data)} records")
            return result
            
        except Exception as e:
            result = {'success': False, 'error': str(e)}
            print(f"✗ Failed to update NG underground storage: {e}")
            return result

    async def update_consumption_percentages_async(self, start=None, end=None, max_concurrent=3):
        """
        Update natural gas consumption percentage data asynchronously.
        
        Args:
            start: Start date in YYYY-MM format
            end: End date in YYYY-MM format  
            max_concurrent: Maximum concurrent requests
            
        Returns:
            Dictionary with update results
        """
        if not start:
            start = "2001-01"
        if not end:
            end = "2025-08"
        
        helper = self.clients["NG"]
        
        try:
            # Update both residential and utility percentages concurrently
            import asyncio
            
            residential_task = helper.get_pct_of_residential_async(start, end, max_concurrent)
            utility_task = helper.get_pct_of_utility_async(start, end, max_concurrent)
            
            residential_data, utility_data = await asyncio.gather(
                residential_task, utility_task, return_exceptions=True
            )
            
            results = {}
            
            # Handle residential data
            if not isinstance(residential_data, Exception):
                residential_data.to_hdf(self.table_db, "NG/consumption/pct_residential")
                results['residential'] = {
                    'success': True,
                    'records': len(residential_data),
                    'key': 'NG/consumption/pct_residential'
                }
                print(f"✓ Updated NG residential percentage: {len(residential_data)} records")
            else:
                results['residential'] = {'success': False, 'error': str(residential_data)}
                print(f"✗ Failed to update NG residential percentage: {residential_data}")
            
            # Handle utility data
            if not isinstance(utility_data, Exception):
                utility_data.to_hdf(self.table_db, "NG/consumption/pct_utility")
                results['utility'] = {
                    'success': True,
                    'records': len(utility_data),
                    'key': 'NG/consumption/pct_utility'
                }
                print(f"✓ Updated NG utility percentage: {len(utility_data)} records")
            else:
                results['utility'] = {'success': False, 'error': str(utility_data)}
                print(f"✗ Failed to update NG utility percentage: {utility_data}")
            
            return results
            
        except Exception as e:
            result = {'success': False, 'error': str(e)}
            print(f"✗ Failed to update NG consumption percentages: {e}")
            return result

    async def bulk_update_natural_gas_async(self, start=None, end=None, max_concurrent=3):
        """
        Update all natural gas data types using NatGasHelper bulk operations.
        
        Args:
            start: Start date in YYYY-MM format
            end: End date in YYYY-MM format
            max_concurrent: Maximum concurrent requests
            
        Returns:
            Dictionary with update results for all data types
        """
        if not start:
            start = "2001-01"
        if not end:
            end = "2025-08"
        
        helper = self.clients["NG"]
        
        try:
            print(f"Starting bulk NG data update for period {start} to {end}")
            start_time = pd.Timestamp.now()
            
            # Use the bulk update method from NatGasHelper
            bulk_results = await helper.bulk_update_all_params_async(start, end, max_concurrent)
            
            # Save the data to HDF5 tables
            stored_keys = []
            failed_keys = []
            
            for param_key, result in bulk_results.items():
                if result['success']:
                    data = result['data']
                    # Map parameter keys to table keys
                    table_key_mapping = {
                        'underground_storage': 'NG/storage/underground',
                        'pct_of_residential': 'NG/consumption/pct_residential',
                        'pct_of_utility': 'NG/consumption/pct_utility'
                    }
                    
                    table_key = table_key_mapping.get(param_key, f"NG/{param_key}")
                    
                    try:
                        data.to_hdf(self.table_db, table_key)
                        stored_keys.append(table_key)
                        print(f"✓ Stored {len(data)} rows to {table_key}")
                    except Exception as save_error:
                        failed_keys.append(param_key)
                        print(f"✗ Failed to save {param_key}: {save_error}")
                else:
                    failed_keys.append(param_key)
            
            end_time = pd.Timestamp.now()
            execution_time = str(end_time - start_time)
            
            final_results = {
                'bulk_results': bulk_results,
                'stored_keys': stored_keys,
                'failed_keys': failed_keys,
                'execution_time': execution_time,
                'successful_updates': len(stored_keys),
                'failed_updates': len(failed_keys)
            }
            
            print(f"✓ Bulk NG update completed in {execution_time}")
            print(f"✓ Successfully updated: {len(stored_keys)} datasets")
            if failed_keys:
                print(f"✗ Failed to update: {len(failed_keys)} datasets")
            
            return final_results
            
        except Exception as e:
            print(f"Fatal error in bulk NG data update: {e}")
            return {'success': False, 'error': str(e)}

    def bulk_update_natural_gas_sync(self, start=None, end=None, max_concurrent=3):
        """
        Synchronous wrapper for bulk_update_natural_gas_async.
        
        Args:
            start: Start date in YYYY-MM format
            end: End date in YYYY-MM format
            max_concurrent: Maximum concurrent requests
            
        Returns:
            Dictionary with update results
        """
        import asyncio
        
        # Check if we're already in an event loop
        try:
            loop = asyncio.get_running_loop()
            # We're already in an async context, need to create a new task
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor() as executor:
                future = executor.submit(
                    asyncio.run,
                    self.bulk_update_natural_gas_async(start, end, max_concurrent)
                )
                return future.result()
        except RuntimeError:
            # No running loop, safe to use asyncio.run
            return asyncio.run(
                self.bulk_update_natural_gas_async(start, end, max_concurrent)
            )

    # Sync wrapper methods
    def update_underground_storage_sync(self, start=None, end=None, max_concurrent=3):
        """Synchronous wrapper for update_underground_storage_async."""
        import asyncio
        
        try:
            loop = asyncio.get_running_loop()
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor() as executor:
                future = executor.submit(
                    asyncio.run,
                    self.update_underground_storage_async(start, end, max_concurrent)
                )
                return future.result()
        except RuntimeError:
            return asyncio.run(
                self.update_underground_storage_async(start, end, max_concurrent)
            )

    def update_consumption_percentages_sync(self, start=None, end=None, max_concurrent=3):
        """Synchronous wrapper for update_consumption_percentages_async."""
        import asyncio
        
        try:
            loop = asyncio.get_running_loop()
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor() as executor:
                future = executor.submit(
                    asyncio.run,
                    self.update_consumption_percentages_async(start, end, max_concurrent)
                )
                return future.result()
        except RuntimeError:
            return asyncio.run(
                self.update_consumption_percentages_async(start, end, max_concurrent)
            )

    # User requested methods for specific keys
    def update_regional_consumption(self, start=None, end=None, use_async=True):
        """
        Update 'NG/demand/consumption' using NatGasHelper regional_consumption methods.
        
        Args:
            start: Start date in YYYY-MM format
            end: End date in YYYY-MM format
            use_async: Whether to use async version (default: True)
            
        Returns:
            bool: True if successful, False otherwise
        """
        helper = self.clients["NG"]
        
        try:
            if use_async:
                import asyncio
                try:
                    loop = asyncio.get_running_loop()
                    import concurrent.futures
                    with concurrent.futures.ThreadPoolExecutor() as executor:
                        future = executor.submit(
                            asyncio.run,
                            helper.get_regional_data_async('consolidated_consumption')
                        )
                        regional_data = future.result()
                except RuntimeError:
                    regional_data = asyncio.run(helper.get_regional_consumption())
            else:
                regional_data = helper.regional_consumption_breakdown_sync()
            
            # Store the data with 2-level MultiIndex
            table_key = "NG/demand/consumption"
            regional_data.to_hdf(self.table_db, table_key)
            
            print(f"✓ Updated {table_key}: {len(regional_data)} rows with {len(regional_data.columns)} columns")
            print(f"  Regional structure: {list(regional_data.columns.get_level_values(0).unique())}")
            
            return True
            
        except Exception as e:
            print(f"✗ Failed to update regional consumption: {e}")
            return False

    async def update_consumption_percentages_by_district_async(self, start=None, end=None, max_concurrent=3):
        """
        Update pct_of_residential and pct_of_utility data to demand/residential/by_district 
        and demand/utility/by_district using async functions.
        
        Args:
            start: Start date in YYYY-MM format
            end: End date in YYYY-MM format
            max_concurrent: Maximum concurrent requests
            
        Returns:
            Dictionary with update results
        """
        if not start:
            start = "2001-01"
        if not end:
            end = "2025-08"
            
        helper = self.clients["NG"]
        results = {}
        
        try:
            print(f"Updating natural gas consumption percentages by district for {start} to {end}")
            
            # Update residential consumption percentages
            try:
                residential_data = await helper.get_pct_of_residential_async(start, end, max_concurrent)
                
                table_key = "NG/demand/residential/by_district"
                residential_data.to_hdf(self.table_db, table_key)
                
                results['residential'] = {
                    'success': True,
                    'key': table_key,
                    'records': len(residential_data),
                    'data': residential_data
                }
                print(f"✓ Updated {table_key}: {len(residential_data)} records")
                
            except Exception as e:
                results['residential'] = {
                    'success': False,
                    'error': str(e)
                }
                print(f"✗ Failed to update residential percentages: {e}")
            
            # Update utility consumption percentages  
            try:
                utility_data = await helper.get_pct_of_utility_async(start, end, max_concurrent)
                
                table_key = "NG/demand/utility/by_district"
                utility_data.to_hdf(self.table_db, table_key)
                
                results['utility'] = {
                    'success': True,
                    'key': table_key, 
                    'records': len(utility_data),
                    'data': utility_data
                }
                print(f"✓ Updated {table_key}: {len(utility_data)} records")
                
            except Exception as e:
                results['utility'] = {
                    'success': False,
                    'error': str(e)
                }
                print(f"✗ Failed to update utility percentages: {e}")
            
            # Summary
            successful = sum(1 for r in results.values() if r.get('success', False))
            total = len(results)
            print(f"\nCompleted consumption percentages update: {successful}/{total} successful")
            
            return results
            
        except Exception as e:
            error_msg = f"Critical error in consumption percentages update: {str(e)}"
            print(f"💥 {error_msg}")
            return {
                'residential': {'success': False, 'error': error_msg},
                'utility': {'success': False, 'error': error_msg}
            }

    def update_consumption_percentages_by_district_sync(self, start=None, end=None, max_concurrent=3):
        """
        Synchronous wrapper for update_consumption_percentages_by_district_async.
        
        Args:
            start: Start date in YYYY-MM format
            end: End date in YYYY-MM format
            max_concurrent: Maximum concurrent requests
            
        Returns:
            Dictionary with update results
        """
        import asyncio
        
        try:
            loop = asyncio.get_running_loop()
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor() as executor:
                future = executor.submit(
                    asyncio.run,
                    self.update_consumption_percentages_by_district_async(start, end, max_concurrent)
                )
                return future.result()
        except RuntimeError:
            return asyncio.run(
                self.update_consumption_percentages_by_district_async(start, end, max_concurrent)
            )
