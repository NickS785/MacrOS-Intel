import os
from datetime import date, datetime
import asyncio as aio
import pandas as pd
import numpy as np
from dotenv import load_dotenv
from MacrOSINT.config import DOT_ENV
from copy import copy
from MacrOSINT.data.sources.eia.EIA_API import EIAClient, AsyncEIAClient
from typing import List, Tuple


# -----------------------------
# Config
# -----------------------------
class FacetParams:
    def __init__(self, duoarea=None, product=None, process=None, series=None):
        args = [duoarea, product, process, series]
        typed_args = []
        for arg in args:
            if isinstance(arg, str):
                new = [arg]
                typed_args.append(new)
            elif isinstance(arg, list):
                typed_args.append(arg)
            else:
                typed_args.append(arg)

        self.duoarea = typed_args[0]
        self.product = typed_args[1]
        self.process = typed_args[2]
        self.series = typed_args[3]
        self.params = self._build_params()

    def _build_params(self):
        params = {}
        if self.duoarea:
            params.update({'duoarea': self.duoarea})
        if self.product:
            params.update({'product': self.product})
        if self.process:
            params.update({'process': self.process})
        if self.series:
            params.update({'series': self.series})
        return params

    def get_params(self):
        return self.params

    def __call__(self):
        return self.params

    def __len__(self):
        return len(self.params)

    def __iter__(self):
        return iter(self.params.items())


class ClientParams:

    def __init__(self, route: str, facets: FacetParams, frequency=None, columns_col: str | Tuple = None, start=None, end=None, **kwargs):
        if columns_col is None:
            columns_col = 'value'
        if not route.startswith('/'):
            route = f'/{route}'
        if route.endswith('/'):
            route[:1]

        if frequency is None:
            frequency = 'monthly'
        self.req_keys = ['route', 'facets', 'frequency', 'start', 'end', 'data_columns']
        self.clean_keys = ['normalize_to_bbl', 'reset_index', 'columns_col', 'sum_value_totals', 'drop_cols']
        self.params = {
            'route': route,
            'facets': facets.params,
            'columns_col': columns_col,
            'frequency': frequency,
            'normalize_to_bbl': False
        }
        if start:
            self._add_start_params()

        for k in kwargs.keys():
            self.params.update({k:kwargs[k]})

        return



    def __call__(self, *args, **kwargs):
        return self.params

    def __add__(self, other):
        if isinstance(other, dict):
            self.params.update(other)
        elif isinstance(other, Tuple):
            if len(other) > 1:
                self.params.update({other[0]: [other[n] for n in range(len(other))]})

    def request(self):
        return {k:v for k, v in self.params.items() if k in self.req_keys}

    def clean(self):
        return {k:v for k,v in self.params.items() if k in self.clean_keys}

    def update_param(self, key, value):
        self.params.update({key:value})

    def update_clean(self, normalize_to_bbl=False, reset_index=False,drop_cols=None, **kwargs):
        self.params.update({
            'normalize_to_bbl':normalize_to_bbl,
            'reset_index':reset_index
        })
        self.params.update({'drop_cols':drop_cols}) if drop_cols else None
        self.params.update({k:v for k,v in kwargs.items() })



    def update_facets(self,key=None, value=None, key_dict=None):

        facets = self.params['facets']
        if key:
            facets.update({key:value})
        else:
            facets.update(key_dict)

        self.params['facets'] = facets

        return

    def _add_start_params(self, start=None, end=None, frequency=None):
        if frequency:
            self.params['frequency'] = frequency

        if self.params['frequency'] != 'annual':
            if start is None:
                start_string = date(2001, 1, 1).strftime('%Y-%m-%d') if self.params['frequency'] == 'weekly' else date(
                    2001, 1, 1).strftime('%Y-%m')
            elif isinstance(start, date) or isinstance(start, datetime):
                start_string = start.strftime('%Y-%m-%d') if self.params['frequency'] == 'weekly' else start.strftime(
                    '%Y-%m')
            else:
                start_string = start

            if end is None:
                end_string = date.today().strftime('%Y-%m-%d') if self.params[
                                                                      'frequency'] == 'weekly' else date.today().strftime(
                    '%Y-%m')
            elif isinstance(end, date) or isinstance(end, datetime):
                end_string = end.strftime('%Y-%m-%d') if self.params['frequency'] == 'weekly' else end.strftime('%Y-%m')
            else:
                end_string = end

        else:
            if start is not None and len(start) != 4:
                if isinstance(start, date) or isinstance(start, datetime):
                    start_string = start.year.__str__()

                    if isinstance(end, date) or isinstance(end, datetime):
                        end_string = end.year.__str__()
                    else:
                        end_string = date.today().year.__str__()
                else:
                    start_string = "2001"
                    end_string = date.today().year.__str__()

            else:
                start_string, end_string = start, end


        self.params.update(
            {
            'start': start_string,
            'end': end_string
            }
        )
        return

    def _get(self, start, end=None):
        self._add_start_params(start, end)

        return self.request()


load_dotenv(DOT_ENV)
eia = EIAClient()  # reads EIA_TOKEN from .env
# API v2 route for natural gas summary (lsum)
FREQ = "monthly"

# State list (50 + DC)
states = [
    "AL", "AK", "AZ", "AR", "CA", "CO", "CT", "DE", "DC", "FL", "GA", "HI", "ID", "IL", "IN", "IA", "KS", "KY",
    "LA", "ME", "MD", "MA", "MI", "MN", "MS", "MO", "MT", "NE", "NV", "NH", "NJ", "NM", "NY", "NC", "ND", "OH",
    "OK", "OR", "PA", "RI", "SC", "SD", "TN", "TX", "UT", "VT", "VA", "WA", "WV", "WI", "WY"
]

SECTORS = {
    "end_use_total": "N3060",  # Delivered to consumers (total)
    "residential": "N3010",
    "commercial": "N3020",
    "industrial": "N3035",
    "electric_power": "N3045",
}

# Optional: Census regions for aggregation
CENSUS_REGIONS = {
    "Northeast": ["CT", "ME", "MA", "NH", "RI", "VT", "NJ", "NY", "PA"],
    "Midwest": ["IL", "IN", "MI", "OH", "WI", "IA", "KS", "MN", "MO", "NE", "ND", "SD"],
    "South": ["DE", "DC", "FL", "GA", "MD", "NC", "SC", "VA", "WV", "AL", "KY", "MS", "TN", "AR", "LA", "OK", "TX"],
    "West": ["AZ", "CO", "ID", "MT", "NV", "NM", "UT", "WY", "AK", "CA", "HI", "OR", "WA"],
}
COUNTRY_NCODES = {"NCA": 'Canada', "NSA": 'Saudi Arabia', "NIZ": "Iraq", "NMX": "Mexico"}

# Unit conversion factors to barrels (bbl)
UNIT_CONVERSION_TO_BBL = {
    # Volume units
    'thousand barrels': 1000,
    'thousand barrels per day': 1000,
    'thousand barrels/day': 1000,
    'million barrels': 1000000,
    'million barrels per day': 1000000,
    'million barrels/day': 1000000,
    'barrels': 1,
    'barrels per day': 1,
    'barrels/day': 1,

    # Gallons to barrels (1 barrel = 42 gallons)
    'thousand gallons': 1000 / 42,
    'million gallons': 1000000 / 42,
    'gallons': 1 / 42,

    # Cubic feet to barrels (for natural gas - approximate)
    # 1 barrel of oil equivalent ≈ 5,800 cubic feet of natural gas
    'thousand cubic feet': 1000 / 5800,
    'million cubic feet': 1000000 / 5800,
    'billion cubic feet': 1000000000 / 5800,
    'cubic feet': 1 / 5800,

    # Common EIA unit abbreviations
    'mbbl': 1000,  # thousand barrels
    'mmbl': 1000000,  # million barrels
    'mbbl/d': 1000,  # thousand barrels per day
    'mmbl/d': 1000000,  # million barrels per day
    'bbl': 1,  # barrels
    'bbl/d': 1,  # barrels per day
}

API_KEY = os.getenv("EIA_TOKEN")  # or hardcode'
BASE = "https://api.eia.gov/v2/natural-gas/sum/lsum/data/"

consumption_processes = ["VC0", "VIN", "VCS", "VRS", "VEU", "VGT"]

product_definitions = {"raw": ["EPC0"],
                       "refined": ["EPM0F", "EPD0", "EPJK"]}

padd_codes = {"PADD1": "R10", 'PADD2': "R20", "PADD3": "R30", 'PADD4': "R40", 'PADD5': 'R50'}
nat_gas_regions = {"R31": "East Coast", "R32": "Midwest", "R34": "Rocky Mountains", "R33": "South Central",
                   "R35": "Pacific Coast", "R48": "Total lower 48"}

gas_consumers = {
                'East':["SPA","SNY", "SVA", "SCT", "SNC", "SNJ", "SMA", "SMD"],
                'Midwest': ["SOH", "SOK", "SIN", "SIL","SMI", "SIA", "SMN", "SMO"],
                'South': ["SAL", "SGA", "SLA", "SMS", "STX", "SAR", "SSC", "SFL"],
                'West': ['SCA', "SAZ", 'SWA', "SOR", "SND", "SCO", "SNV", "SUT"],
                'Total 50': ["NUS"]
                 }

rev_codes = {v: k for k, v in padd_codes.items()}

padd_states = {
    "PADD 1": [
        "CT", "DE", "DC", "FL", "GA", "ME", "MD", "MA", "NH", "NJ", "NY",
        "NC", "PA", "RI", "SC", "VT", "VA", "WV"
    ],
    "PADD 2": [
        "IL", "IN", "IA", "KS", "KY", "MI", "MN", "MO", "NE", "ND", "OH",
        "OK", "SD", "TN", "WI"
    ],
    "PADD 3": [
        "AL", "AR", "LA", "MS", "NM", "TX"
    ],
    "PADD 4": [
        "CO", "ID", "MT", "UT", "WY"
    ],
    "PADD 5": [
        "AK", "AZ", "CA", "HI", "NV", "OR", "WA"
    ],
}

states_plus_usa = ["NUS"] + [f"S{st}" for st in states]
duoareas_dict = {'states_plus': states_plus_usa, 'offshore': ['R3FM']}

# FIPS codes for all states + DC
STATE_FIPS = {
    "AL": "01", "AK": "02", "AZ": "04", "AR": "05", "CA": "06", "CO": "08", "CT": "09", "DE": "10", "DC": "11",
    "FL": "12",
    "GA": "13", "HI": "15", "ID": "16", "IL": "17", "IN": "18", "IA": "19", "KS": "20", "KY": "21", "LA": "22",
    "ME": "23",
    "MD": "24", "MA": "25", "MI": "26", "MN": "27", "MS": "28", "MO": "29", "MT": "30", "NE": "31", "NV": "32",
    "NH": "33",
    "NJ": "34", "NM": "35", "NY": "36", "NC": "37", "ND": "38", "OH": "39", "OK": "40", "OR": "41", "PA": "42",
    "RI": "44",
    "SC": "45", "SD": "46", "TN": "47", "TX": "48", "UT": "49", "VT": "50", "VA": "51", "WA": "53", "WV": "54",
    "WI": "55", "WY": "56"
}


def _normalize_values_to_bbl(df: pd.DataFrame, value_col: str = 'value') -> pd.DataFrame:
    """
    Normalize values to barrels based on their units.
    
    Args:
        df: DataFrame with 'units' and value columns
        value_col: Name of the value columns_col to normalize
        
    Returns:
        DataFrame with normalized values
    """
    df = df.copy()

    # Group by units to apply conversions efficiently
    if 'units' not in df.columns:
        print("Warning: No 'units' columns_col found for normalization")
        return df

    for unit in df['units'].unique():
        if pd.isna(unit):
            continue

        unit_lower = str(unit).lower().strip()
        conversion_factor = None

        # Find matching conversion factor
        for unit_key, factor in UNIT_CONVERSION_TO_BBL.items():
            if unit_lower == unit_key.lower() or unit_lower in unit_key.lower():
                conversion_factor = factor
                break

        if conversion_factor is not None:
            # Apply conversion to rows with this unit
            mask = df['units'] == unit
            df.loc[mask, value_col] = df.loc[mask, value_col] * conversion_factor
            print(f"Converted {unit} to barrels using factor: {conversion_factor}")
        else:
            print(f"Warning: No conversion factor found for unit '{unit}'. Values left unchanged.")

    return df


def clean_api_data(data, date_col='period', columns_col: Tuple[str] = 'area-name', value_col='value', unit_col="units",
                   sum_value_totals=False, flatten_columns=False, normalize_to_bbl=True, reset_index=True, drop_cols=None):
    """
    Reshape DataFrame to have area names as columns and values as data.

    Steps:
    1. Group rows by date to consolidate data for each time period
    2. Convert data from string to correct dtype
    3. (Optional) Normalize values to barrels based on units
    4. Pivot the DataFrame to make area names become columns_col headers
    5. Use the original value columns_col as the data for the new area columns
    6. Reset index to make date a regular columns_col again
    7. (Optional) Sum up columns area columns for a national total
    8. (Optional) Flatten MultiIndex columns for HDF5 compatibility
    
    Args:
        data: Input data (list of dicts or DataFrame)
        date_col: Name of the date columns_col (default: 'period')
        columns_col: Name of the area columns_col (default: 'area-name')
        value_col: Name of the value columns_col (default: 'value')
        sum_value_totals: Add national/regional total columns (default: False)
        flatten_columns: Flatten MultiIndex columns for HDF5 storage (default: False)
        normalize_to_bbl: Convert values to barrels based on units (default: False)

    Returns:
        DataFrame with dates as rows and area names as columns
        :rtype: pd.DataFrame
    """
    # Convert date columns_col to datetime for proper sorting
    df = pd.DataFrame(data)
    df[date_col] = pd.to_datetime(df[date_col])
    df[value_col] = df[value_col].astype(float)
    units = df[unit_col]

    # Normalize values to barrels if requested
    if normalize_to_bbl and 'units' in df.columns:
        df = _normalize_values_to_bbl(df, value_col)
        # Units will be handled after pivot - all normalized to barrels

    df.drop(columns=['units'], inplace=True
            )

    # Pivot the DataFrame: dates as index, area names as columns, values as data
    pivoted = df.pivot_table(
        index=date_col,
        columns=columns_col,
        values=value_col,
        aggfunc='first'  # Use first value if duplicates exist
    )
    pivoted.sort_index(inplace=True)

    if sum_value_totals:
        if isinstance(columns_col, str):
            # Single-level columns: add overall total
            try:
                pivoted['Total'] = pivoted.sum(axis=1, numeric_only=True)
            except Exception as e:
                print(f'Failed to sum Data.\n Error:{e}')
        else:
            # MultiIndex columns: calculate totals for each top-level category
            if isinstance(pivoted.columns, pd.MultiIndex):
                # Get unique values from the first level (categories)
                level_0_values = pivoted.columns.get_level_values(0).unique()
                
                for category in level_0_values:
                    try:
                        # Get all columns for this category
                        category_columns = pivoted[category]

                        # Sum numeric columns only for this category
                        category_total = category_columns.sum(axis=1)

                        # Add the total as a new column in the MultiIndex
                        pivoted[(category, 'Total')] = category_total
                        
                    except Exception as e:
                        print(f'Failed to calculate total for category "{category}": {e}')
                
                # Optionally add grand total across all categories
                try:
                    # Sum all numeric data across all categories
                    grand_total = pivoted.select_dtypes(include=[np.number]).sum(axis=1, numeric_only=True)
                    pivoted[('Grand Total', 'All Categories')] = grand_total
                except Exception as e:
                    print(f'Failed to calculate grand total: {e}')
            else:
                # Fallback for non-MultiIndex but multi-column case
                try:
                    pivoted['Total'] = pivoted.sum(axis=1, numeric_only=True)
                except Exception as e:
                    print(f'Failed to sum data for multi-column case: {e}')


    # Reset index to make date a regular columns_col
    if reset_index:
        pivoted.reset_index(inplace=True)

    # Add units columns_col based on normalization setting
    try:
        if normalize_to_bbl:
            # All data normalized to barrels
            pivoted['units'] = 'barrels'
        else:
            # Get representative unit value from original data
            # Since all data in one API call typically has same units, use the first unique unit value
            unique_units = units.unique()
            unit_value = unique_units[0] if len(unique_units) > 0 else 'unknown'
            pivoted['units'] = units
    except Exception as e:
        print(f"Warning: Could not assign units: {e}")
        pivoted['units'] = 'unknown'

    # Clean up columns_col names (remove the area columns_col name from headers)
    pivoted.columns.name = None

    # Flatten MultiIndex columns if requested (for HDF5 compatibility)
    if drop_cols:
        pivoted.drop(columns=drop_cols, inplace=True)

    return pivoted


def aggregate_regions(wide_states_df, region_map=None, sum_columns=None, date_column="date"):
    """
    Args:
        wide_states_df: dataframe containing values and a state columns_col
        region_map: Dictionary containing region and state names
        sum_columns: value columns selected to add
        date_column: Column which groups columns together
    """
    # Wide by state, then sum per region
    if region_map is None:
        region_map = CENSUS_REGIONS
    if sum_columns is None:
        sum_columns = ["Delivered"]
    reg_frames = []
    for region, st_list in region_map.items():
        reg = (wide_states_df[wide_states_df['state'].isin(st_list)].
        groupby(date_column)[sum_columns].sum().
        copy().rename({
            col: "".join([region, " ", col]) for col in sum_columns
        }, axis=1))
        reg_frames.append(reg)

    return pd.concat(reg_frames, axis=1)


class NatGasHelper:

    def __init__(self):
        self.routes = {'summary': ("sum", "lsum"), 'offshore_production': ('prod', 'off'), 'production': ('prod')}
        self.regions = [*nat_gas_regions.keys()]
        self.PARAMS = {

            'underground_storage': ClientParams(
                route='/stor/wkly',
                columns_col='duoarea',
                frequency='weekly',
                facets=FacetParams(
                    duoarea=self.regions,
                    process=["SWO"],
                )),
            'spot_prices': ClientParams(
                route='/pri/fut',
                columns_col='product',
                frequency='daily',
                facets=FacetParams(
                    process=["PS0"]
                )
            ),

            'state_pct_of_consumption': ClientParams(
                route='/cons/pns',
                columns_col=('process', 'duoarea'),
                frequency='annual',
                facets=FacetParams(
                    duoarea=states_plus_usa,
                    process=["VRP", "VEP"],
                )),

            'consolidated_consumption':ClientParams(
                route='/cons/sum',
                columns_col=('process','duoarea'),
                frequency='monthly',
                facets=FacetParams(
                    process=["VRS", "VEU", "VGT"]
                )),
            'production':ClientParams(
                route='/prod/sum',
                columns_col='duoarea',
                frequency='monthly',
                facets=FacetParams(
                    process=['FGW'],
                    duoarea=states_plus_usa
                )),

            'state_production_detailed': ClientParams(
                route='/prod/sum',
                columns_col='duoarea',
                frequency='monthly',
                facets=FacetParams(
                    process=['FGW'],
                    duoarea=[
                        'NUS', 'R3FM', 'R98', 'SAK', 'SAL', 'SAR', 'SAZ', 'SCA', 'SCO', 'SFL',
                        'SID', 'SIL', 'SIN', 'SKS', 'SKY', 'SLA', 'SMD', 'SMI', 'SMO', 'SMS',
                        'SMT', 'SND', 'SNE', 'SNM', 'SNV', 'SNY', 'SOH', 'SOK', 'SOR', 'SPA',
                        'SSD', 'STN', 'STX', 'SUT', 'SVA', 'SWV', 'SWY'
                    ]
                )),

            # Exports: LNG (ENG) and pipeline (ENP) by border crossing
            # YSPL-Z00 = Sabine Pass LNG terminal (largest single feedgas sink)
            'lng_exports': ClientParams(
                route='/move/poe2',
                columns_col='duoarea',
                frequency='monthly',
                facets=FacetParams(
                    process=['ENG'],
                    duoarea=['NUS-NCA', 'NUS-NMX', 'NUS-Z00', 'YSPL-Z00'],
                )),

            # Sabine Pass LNG terminal only — highest-volatility single export point
            'sabine_pass_exports': ClientParams(
                route='/move/poe2',
                columns_col='duoarea',
                frequency='monthly',
                facets=FacetParams(
                    process=['ENG'],
                    duoarea=['YSPL-Z00'],
                )),

            # Canadian pipeline imports — dominant import source, strong winter signal
            'canada_pipeline_imports': ClientParams(
                route='/move/poe1',
                columns_col='duoarea',
                frequency='monthly',
                facets=FacetParams(
                    process=['IRP'],
                    duoarea=['NUS-NCA'],
                )),

            'pipeline_exports': ClientParams(
                route='/move/poe2',
                columns_col='duoarea',
                frequency='monthly',
                facets=FacetParams(
                    process=['ENP'],
                    duoarea=['NUS-NCA', 'NUS-NMX', 'NUS-Z00'],
                )),

            # Combined LNG + pipeline exports
            'total_exports': ClientParams(
                route='/move/poe2',
                columns_col=('process', 'duoarea'),
                frequency='monthly',
                facets=FacetParams(
                    process=['ENG', 'ENP'],
                    duoarea=['NUS-NCA', 'NUS-NMX', 'NUS-Z00'],
                )),

            # Imports: pipeline (IRP) and LNG (IML) by border crossing
            'pipeline_imports': ClientParams(
                route='/move/poe1',
                columns_col='duoarea',
                frequency='monthly',
                facets=FacetParams(
                    process=['IRP'],
                    duoarea=['NUS-NCA', 'NUS-NMX', 'NUS-Z00'],
                )),

            'lng_imports': ClientParams(
                route='/move/poe1',
                columns_col='duoarea',
                frequency='monthly',
                facets=FacetParams(
                    process=['IML'],
                    duoarea=['NUS-NCA', 'NUS-NMX', 'NUS-Z00'],
                )),

            # Combined pipeline + LNG imports
            'total_imports': ClientParams(
                route='/move/poe1',
                columns_col=('process', 'duoarea'),
                frequency='monthly',
                facets=FacetParams(
                    process=['IRP', 'IML'],
                    duoarea=['NUS-NCA', 'NUS-NMX', 'NUS-Z00'],
                )),

        }
        self.top_producers = {
            "STX": "Texas",
            "SPA": "Pennsylvania",
            "SLA": "Louisiana",
            "SWV": "West Virginia",
            "SNM": "New Mexico",
            "SOK": "Oklahoma",
            "SCO": "Colorado",
            "SOH": "Ohio",
            "SWY": "Wyoming",
            "SND": "North Dakota",
            "OFF": "Federal Offshore - Gulf of Mexico"
        }

        self.client = EIAClient(os.getenv('EIA_TOKEN'), timeout=120).natural_gas
        self.async_client_config = {
            'api_key': os.getenv('EIA_TOKEN'),
            'route': '/natural-gas',
            'max_concurrent': 5
        }
        self.async_client = AsyncEIAClient(**self.async_client_config)

        return


    def execute_request(self, param_key: str, start=None, end=None, **kwargs):
        """
        Main request function that uses ClientParams configurations from self.PARAMS.

        Args:
            param_key: Key from self.PARAMS to use for request configuration
            start: Start date override
            end: End date override
            **kwargs: Additional parameters to override ClientParams settings

        Returns:
            DataFrame with cleaned API data
        """
        if param_key not in self.PARAMS:
            raise ValueError(f"Parameter key '{param_key}' not found in self.PARAMS. Available keys: {list(self.PARAMS.keys())}")

        # Get the ClientParams configuration
        client_params = self.PARAMS[param_key]

        # Override with provided parameters
        if start or end:
            client_params._add_start_params(start, end, kwargs.get('frequency'))

        # Get request parameters using the new interface
        request_params = client_params.request()
        request_params.update(kwargs)  # Override any other parameters

        # Execute the request - extract route as positional arg
        route = request_params.pop('route')
        data = self.client.get_all_data(route, **request_params)


        # Clean and return the data using ClientParams.clean()
        clean_params = client_params.clean()

        return clean_api_data(data, **clean_params)

    async def execute_request_async(self, param_key: str, start=None, end=None, max_concurrent=5, **kwargs):
        """
        Async version of execute_request for improved performance.

        Args:
            param_key: Key from self.PARAMS to use for request configuration
            start: Start date override
            end: End date override
            max_concurrent: Maximum concurrent requests
            **kwargs: Additional parameters to override ClientParams settings

        Returns:
            DataFrame with cleaned API data
        """
        if param_key not in self.PARAMS:
            raise ValueError(f"Parameter key '{param_key}' not found in self.PARAMS. Available keys: {list(self.PARAMS.keys())}")

        # Get the ClientParams configuration
        client_params = self.PARAMS[param_key]

        # Override with provided parameters
        if start or end:
            client_params._add_start_params(start, end)

        # Get request parameters using the new interface
        request_params = client_params.request()
        request_params.update(kwargs)  # Override any other parameters

        #
        # Execute the async request using AsyncEIAClient as context manager
        try:
            # Extract route from request_params (AsyncEIAClient expects route as first arg)
            # Create async client with context manager
            async_client = AsyncEIAClient(
                **self.async_client_config
            )

            async with self.async_client as client:
                if hasattr(client, 'get_all_data_async'):
                    route = request_params.pop('route')
                    data = await client.get_all_data_async(route, **request_params)
                else:
                    # Fallback to sync method if async not available
                    print(f"Warning: Async client doesn't support get_all_data_async, falling back to sync for {param_key}")
                    # Restore route to request_params for sync call
                    data = self.client.get_all_data(**request_params)

        except Exception as e:
            print(f"Error in async request for {param_key}: {e}")
            # Fallback to sync client - need to restore route if it was pop
            data = self.client.get_all_data(**request_params)

        # Clean and return the data using ClientParams.clean()
        clean_params = client_params.clean()
        return clean_api_data(data, **clean_params)

    async def get_state_production_detailed_async(self, start=None, end=None, max_concurrent=5):
        """
        Get detailed state-level natural gas production data using execute_request_async.
        
        This method uses the 'state_production_detailed' parameter configuration that includes
        all states plus offshore regions and federal areas, corresponding to the URL:
        https://api.eia.gov/v2/natural-gas/prod/sum/data/?frequency=monthly&data[0]=value&facets[duoarea][]=...
        
        Args:
            start: Start date in YYYY-MM format (default: None)
            end: End date in YYYY-MM format (default: None)  
            max_concurrent: Maximum concurrent requests (default: 5)
        
        Returns:
            DataFrame with state-level production data, columns represent duoarea codes
        """
        return await self.execute_request_async(
            param_key='state_production_detailed',
            start=start,
            end=end,
            max_concurrent=max_concurrent
        )

    async def get_regional_data_async(self, facet_key,start=None,end=None, max_concurrent=5, regions=None, sum_totals=False,reset_indexes=False, **kwargs):
        dfs = {}
        if not regions:
            regions = gas_consumers

        CParams = self.PARAMS[facet_key]

        for k, v in regions.items():
            CParams.update_facets('duoarea', v)
            req_params = CParams.request()
            route = req_params.pop('route')
            async with self.async_client as client:
                try:
                    data = await client.get_all_data_async(route, **req_params)
                    CParams.update_clean(drop_cols=['units'], reset_indexes=False, sum_all_values=sum_totals, **kwargs)
                    dfs[k] = clean_api_data(data, **CParams.clean())
                except:
                    print(f'Error getting region {k}')

        master = pd.concat(dfs, axis=1,keys=regions.keys())

        return master
    async def get_spot_prices_async(self, start=None, end=None, max_concurrent=5):
        return await self.execute_request_async('spot_prices', start=start, end=end, max_concurrent=max_concurrent)

    def get_spot_prices(self, start=None, end=None):
        data = aio.run(self.execute_request_async('spot_prices', start=start, end=end, max_concurrent=5))

        return data


    # Async methods for all parameters in self.PARAMS
    async def get_underground_storage_async(self, start=None, end=None, max_concurrent=5):
        """Get underground natural gas storage data asynchronously."""
        return await self.execute_request_async('underground_storage', start, end, max_concurrent)


    async def get_state_consumption_as_pct_async(self, start=None, end=None, max_concurrent=5):
        """Get percentage of residential natural gas consumption data asynchronously."""
        return await self.execute_request_async('state_pct_of_consumption', start, end, max_concurrent)


    # Sync wrapper methods for all parameters in self.PARAMS
    def get_underground_storage(self, start=None, end=None):
        """Get underground natural gas storage data."""
        return self.execute_request('underground_storage', start, end)

    def get_pct_of_residential(self, start=None, end=None):
        """Get percentage of residential natural gas consumption data."""
        return self.execute_request('pct_of_residential', start, end)

    def get_pct_of_utility(self, start=None, end=None):
        """Get percentage of utility natural gas consumption data."""
        return self.execute_request('pct_of_utility', start, end)

    def get_lng_exports(self, start=None, end=None):
        """Get US LNG exports by border crossing (Canada/Mexico/Other)."""
        return self.execute_request('lng_exports', start, end)

    def get_pipeline_exports(self, start=None, end=None):
        """Get US pipeline natural gas exports by border crossing."""
        return self.execute_request('pipeline_exports', start, end)

    def get_total_exports(self, start=None, end=None):
        """Get US natural gas total exports (LNG + pipeline) by process and border crossing."""
        return self.execute_request('total_exports', start, end)

    def get_pipeline_imports(self, start=None, end=None):
        """Get US pipeline natural gas imports by border crossing."""
        return self.execute_request('pipeline_imports', start, end)

    def get_lng_imports(self, start=None, end=None):
        """Get US LNG imports by border crossing."""
        return self.execute_request('lng_imports', start, end)

    def get_total_imports(self, start=None, end=None):
        """Get US natural gas total imports (pipeline + LNG) by process and border crossing."""
        return self.execute_request('total_imports', start, end)

    async def get_lng_exports_async(self, start=None, end=None, max_concurrent=5):
        """Get US LNG exports asynchronously."""
        return await self.execute_request_async('lng_exports', start, end, max_concurrent)

    async def get_pipeline_exports_async(self, start=None, end=None, max_concurrent=5):
        """Get US pipeline exports asynchronously."""
        return await self.execute_request_async('pipeline_exports', start, end, max_concurrent)

    async def get_total_exports_async(self, start=None, end=None, max_concurrent=5):
        """Get US total natural gas exports asynchronously."""
        return await self.execute_request_async('total_exports', start, end, max_concurrent)

    async def get_pipeline_imports_async(self, start=None, end=None, max_concurrent=5):
        """Get US pipeline imports asynchronously."""
        return await self.execute_request_async('pipeline_imports', start, end, max_concurrent)

    async def get_lng_imports_async(self, start=None, end=None, max_concurrent=5):
        """Get US LNG imports asynchronously."""
        return await self.execute_request_async('lng_imports', start, end, max_concurrent)

    async def get_total_imports_async(self, start=None, end=None, max_concurrent=5):
        """Get US total natural gas imports asynchronously."""
        return await self.execute_request_async('total_imports', start, end, max_concurrent)

    def get_sabine_pass_exports(self, start=None, end=None):
        """Get Sabine Pass LNG terminal exports (YSPL-Z00)."""
        return self.execute_request('sabine_pass_exports', start, end)

    async def get_sabine_pass_exports_async(self, start=None, end=None, max_concurrent=5):
        """Get Sabine Pass LNG terminal exports asynchronously."""
        return await self.execute_request_async('sabine_pass_exports', start, end, max_concurrent)

    def get_canada_pipeline_imports(self, start=None, end=None):
        """Get Canadian pipeline imports into the US (NUS-NCA, process IRP)."""
        return self.execute_request('canada_pipeline_imports', start, end)

    async def get_canada_pipeline_imports_async(self, start=None, end=None, max_concurrent=5):
        """Get Canadian pipeline imports asynchronously."""
        return await self.execute_request_async('canada_pipeline_imports', start, end, max_concurrent)

    async def get_regional_consumption(self):

        async def get_region_async(region_name=None):
            CParams = self.PARAMS['consolidated_consumption']
            if not region_name:
                region_name = "East"

            CParams.update_facets('duoarea', gas_consumers[region_name])

            # Extract route for async client
            request_params = CParams.request()
            route = request_params.pop('route', '/cons/sum')

            async with self.async_client as client:
                data = await client.get_all_data_async(route, **request_params)

            CParams.update_clean(sum_value_totals=True)
            df = clean_api_data(data, **CParams.clean())

            return df
        # Collect DataFrames for each region
        region_dfs = {}
        
        for region in gas_consumers.keys():
            df = await get_region_async(region)
            region_dfs[region] = df

        # Create proper 2-level MultiIndex using pd.concat with keys
        master = pd.concat(region_dfs, axis=1, keys=region_dfs.keys())
        
        return master

    def regional_consumption_sync(self, start=None, end=None):
        """
        Synchronous version of regional_consumption_breakdown with 2-level MultiIndex.
        """
        def get_region_sync(region_name=None):
            CParams = self.PARAMS['consolidated_consumption']
            if not region_name:
                region_name = "East"

            CParams.update_facets('duoarea', gas_consumers[region_name])
            if start or end:
                CParams._add_start_params(start, end)

            # Extract route for sync client
            request_params = CParams.request()
            route = request_params.pop('route')

            # Execute sync request
            data = self.client.get_all_data(route, **request_params)

            CParams.update_clean(
                sum_value_totals=True
            )
            df = clean_api_data(data, **CParams.clean())

            return df

        # Collect DataFrames for each region
        region_dfs = {}
        
        for region in gas_consumers.keys():
            df = get_region_sync(region)
            region_dfs[region] = df


        # Create proper 2-level MultiIndex using pd.concat with keys
        master = pd.concat(region_dfs, axis=1, keys=region_dfs.keys())

        return master


    def get_state_consumption_as_pct(self, start=None, end=None):
        """
        Update all consumption-related parameters synchronously.
        
        Args:
            start: Start date for all requests
            end: End date for all requests
            
        Returns:
            Dictionary with results for each parameter key
        """
        # Parameters related to consumption
        consumption_params = [
            'pct_of_residential', 
            'pct_of_utility',
        ]
        
        results = {}
        
        for param_key in consumption_params:
            if param_key in self.PARAMS:
                try:
                    print(f"Updating {param_key}...")
                    result = self.execute_request(param_key, start, end)
                    results[param_key] = {
                        'success': True,
                        'data': result,
                        'error': None
                    }
                    print(f"✓ Successfully updated {param_key}")
                    results['data']
                except Exception as e:
                    print(f"✗ Failed to update {param_key}: {e}")
                    results[param_key] = {
                        'success': False,
                        'data': None,
                        'error': str(e)
                    }
        
        # Print summary
        successful = sum(1 for r in results.values() if r['success'])
        total = len(results)
        print(f"\nSync update completed: {successful}/{total} successful")
        
        return results

    def get_production_data(self, start=None, end=None):

        return


    # Bulk async operations
    async def bulk_update_all_params_async(self, start=None, end=None, max_concurrent=3):
        """
        Update all parameters from self.PARAMS concurrently.
        
        Args:
            start: Start date for all requests
            end: End date for all requests  
            max_concurrent: Maximum concurrent requests
            
        Returns:
            Dictionary with results for each parameter key
        """
        import asyncio

        # Create tasks for all parameters
        tasks = {}
        for param_key in self.PARAMS.keys():
            tasks[param_key] = self.execute_request_async(param_key, start, end, max_concurrent)
        
        # Execute all tasks concurrently
        results = {}
        try:
            completed_tasks = await asyncio.gather(*tasks.values(), return_exceptions=True)
            
            # Map results back to parameter keys
            for i, (param_key, task_result) in enumerate(zip(tasks.keys(), completed_tasks)):
                if isinstance(task_result, Exception):
                    results[param_key] = {'success': False, 'error': str(task_result)}
                    print(f"✗ Failed to update {param_key}: {task_result}")
                else:
                    results[param_key] = {'success': True, 'data': task_result, 'records': len(task_result)}
                    print(f"✓ Updated {param_key}: {len(task_result)} records")
                    
        except Exception as e:
            print(f"Fatal error in bulk update: {e}")
            raise
            
        return results

    def bulk_update_all_params_sync(self, start=None, end=None, max_concurrent=3):
        """
        Synchronous wrapper for bulk_update_all_params_async.
        
        Args:
            start: Start date for all requests
            end: End date for all requests
            max_concurrent: Maximum concurrent requests
            
        Returns:
            Dictionary with results for each parameter key
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
                    self.bulk_update_all_params_async(start, end, max_concurrent)
                )
                return future.result()
        except RuntimeError:
            # No running loop, safe to use asyncio.run
            return asyncio.run(
                self.bulk_update_all_params_async(start, end, max_concurrent)
            )


# noinspection PyTypeChecker
class PetroleumHelper:

    def __init__(self):

        self.agg_level = {
            'National': ['NUS'],
            'PADD': ['R10', 'R20', 'R30', 'R40', 'R50'],
            'top_exporter_codes': ["NMX", "NIZ", "NCA", "NSA"]
        }

        self.FACET_PARAMS = {
            'receipts_from_producers': FacetParams(
                product=['EPC0'],
                process=["TNR"],
                series=["MCRIPP11", "MCRIPP1MX1",
                        "MCRIPP2CA1", "MCRIPP2MX1",
                        "MCRIPP2SA1", "MCRIPP3CA1",
                        "MCRIPP3MX1", "MCRIPP3SA1", "MCRIPP41",
                        "MCRIPP4CA1", "MCRIPP51", "MCRIPP5MX1", "MCRIPP5SA1",
                        "MCRIP_R10-NBX_1", "MCRIP_R10-NIZ_1", "MCRIP_R20-NIZ_1",
                        "MCRIP_R30-NIZ_1", "MCRIP_R40-NMX_1", "MCRIP_R40-NSA_1",
                        "MCRIP_R50-NIZ_1"]).params,
            'spot_prices': FacetParams(duoarea=['RGC', 'YCUOK'],
                                       product=["EPCWTI","EPD2DXL0","EPJK","EPMRU"],
                                       process=["PF4"]).params,

            'imports_to_padd': FacetParams(
                duoarea=["R10-NCA", "R10-NBX", "R10-NIZ", "R10-NMX", "R10-Z00", "R20-NCA",
                         "R20-NIZ", "R20-NMX", "R20-NSA", "R20-Z00", "R30-NCA",
                         "R30-NIZ", "R30-NMX", "R30-NSA", "R30-Z00", "R40-NCA",
                         "R40-NIZ", "R40-NMX", "R40-NSA", "R40-Z00", "R50-NIZ",
                         "R50-NMX", "R50-NSA", "R50-Z00"],
                process=["IP0"],
                product=["EPC0"],
                series=["MCRIMP11", "MCRIMP21", "MCRIMP31", "MCRIMP41",
                        "MCRIMP51", "MCRIPP11", "MCRIPP1MX1", "MCRIPP21",
                        "MCRIPP2CA1", "MCRIPP2MX1", "MCRIPP2SA1",
                        "MCRIPP31", "MCRIPP3CA1", "MCRIPP3MX1",
                        "MCRIPP3SA1", "MCRIPP41", "MCRIPP4CA1",
                        "MCRIPP51", "MCRIPP5MX1", "MCRIPP5SA1",
                        "MCRIP_R10-NBX_1", "MCRIP_R10-NIZ_1", "MCRIP_R20-NIZ_1",
                        "MCRIP_R30-NIZ_1", "MCRIP_R40-NMX_1", "MCRIP_R40-NSA_1",
                        "MCRIP_R50-NIZ_1"]
            ).params,

            'exports_from_padd': FacetParams(
                duoarea=["EPC0", "EPD0", "EPJK", "EPM0F", "EPOOXE"],
                process=["EEX"],
                series=["MCREXP11", "MCREXP21", "MCREXP31", "MCREXP41", "MCREXP51",
                        "MCREXUS1", "MDIEXP11", "MDIEXP21", "MDIEXP31", "MDIEXP41",
                        "MDIEXP51", "MDIEXUS1", "MGFEXP11", "MGFEXP21", "MGFEXP31",
                        "MGFEXP41", "MGFEXP51", "MGFEXUS1", "MKJEXP11", "MKJEXP21",
                        "MKJEXP31", "MKJEXP41", "MKJEXP51", "MKJEXUS1", "M_EPOOXE_EEX_NUS-Z00_MBBL",
                        "M_EPOOXE_EEX_R10-Z00_MBBL", "M_EPOOXE_EEX_R20-Z00_MBBL", "M_EPOOXE_EEX_R30-Z00_MBBL"
                    , "M_EPOOXE_EEX_R40-Z00_MBBL", "M_EPOOXE_EEX_R50-Z00_MBBL"]
            ).params,

            'imports': FacetParams(
                duoarea=["".join([dist, "-Z00"]) for dist in self.agg_level["PADD"] + ["NUS"]],
                process=["IM0"],
                product=["EPC0", "EPM0F", "EPD0", "EPJK"]

            ).params,
            'exports': FacetParams(
                duoarea=["".join([dist, "-Z00"]) for dist in self.agg_level["PADD"] + ["NUS"]],
                process=["EEX"],
                product=["EPC0", "EPM0F", "EPD0", "EPJK"],
                series=["MCREXP11", "MCREXP21", "MCREXP31", "MCREXP41",
                        "MCREXP51", "MDIEXP11", "MDIEXP21", "MDIEXP31",
                        "MDIEXP41", "MDIEXP51", "MGFEXP11", "MGFEXP21",
                        "MGFEXP31", "MGFEXP41", "MGFEXP51", "MKJEXP11",
                        "MKJEXP21", "MKJEXP31", "MKJEXP41", "MKJEXP51"]

            ).params,
            'product_stocks': FacetParams(
                duoarea=self.agg_level['PADD'] + ["NUS"],
                process=["SAE"],
                product=["EPC0", "EPM0F", "EPD0", "EPJK"]

            ).params,
            'crude_movements': FacetParams(
                duoarea=["".join([dist, "-Z0P"]) for dist in self.agg_level['PADD']],
                process=["TNR"],  # Total for crude oil movements
                product=["EPC0"],  # Crude oil only
                series=[
                    "MCRMXP1P21",
                    "MCRMXP1P31",
                    "MCRMXP1P51",
                    "MCRMXP2P11",
                    "MCRMXP2P31",
                    "MCRMXP2P41",
                    "MCRMXP3P11",
                    "MCRMXP3P21",
                    "MCRMXP3P41",
                    "MCRMXP3P51",
                    "MCRMXP4P21",
                    "MCRMXP5P31",
                    "MCRMX_R20-R50_1",
                    "MCRMX_R40-R30_1",
                    "MCRMX_R40-R50_1",
                    "MCRMX_R50-R20_1",
                    "MCRMX_R50-R40_1",
                    "M_EPC0_TNR_R10-R40_1",
                    "M_EPC0_TNR_R40-R10_1"
                ]
            ).params,
            'refined_product_movements': FacetParams(
                duoarea=["".join([dist, "-Z0P"]) for dist in self.agg_level['PADD']],
                process=["VNR"],  # Net Receipts for refined product movements
                product=["EPD0", "EPJK", "EPM0F"]  # Refined products only
            ).params,
            'tank_farm_stocks': FacetParams(
                duoarea=self.agg_level['PADD'],
                process=["STT"]
            ).params,
            'refinery_stocks': FacetParams(
                duoarea=self.agg_level['PADD'] + ["NUS"],
                process=["SKR"],
                product=['EPC0']
            ).params,
            'refined_stocks': FacetParams(
                duoarea=self.agg_level['PADD'] + ["NUS"],
                process=['SAE'],
                product=['EPM0', 'EPD0']
            ).params,
            'refinery_production': FacetParams(
                duoarea=self.agg_level['PADD'] + ["NUS"],
                process=["YPR"],
                product=["EPM0F", "EPD0", "EPJK"]
            ).params,
            'consumption_breakdown': FacetParams(
                duoarea=self.agg_level['National'],
                process=['VPP'],
                product=["EPD0", "EPJK", "EPLLPZ", "EPM0F", "EPPO4"]
            ).params,
            'crude_production': FacetParams(
                duoarea=self.agg_level['PADD'],
                process=["FPF"],
                product=['EPC0']
            ).params,
            'refinery_consumption': FacetParams(
                duoarea=self.agg_level['PADD'] + ["NUS"],
                process=["YIY"],
                product=["EPC0"]
            ).params,
            'product_supplied': FacetParams(
                duoarea=self.agg_level['National'] + ['NUS'],
                process=["VPP"],
                product=["EPC0","EPM0F", "EPD0", "EPJK"]
            ).params,
            'refinery_utilization': FacetParams(
                duoarea=self.agg_level['PADD'] + ['NUS'],
                process=['YUP', 'YRL']
            ).params
        }
        self.stocks_client = eia.petroleum.stoc
        self.consumption_client = eia.petroleum.cons
        self.client = eia.petroleum
        self.async_client = AsyncEIAClient(max_concurrent=5, route='/petroleum')

        return

    def execute_request(self, route, facet_key, columns_col="area-name", sum_values=False, frequency="monthly",
                        start=None, end=None):
        if not start:
            start = date(2000, 1, 1).strftime('%Y-%m-%d')
            end = date.today().strftime('%Y-%m-%d')
        if frequency == 'monthly':
            start, end = start[:-3], end[:-3]
        elif frequency == 'annual':
            start, end = start[:4], end[:4]
            
        data = self.client.get_all_data(route,
                                        data_columns=["value"],
                                        facets=self.FACET_PARAMS[facet_key],
                                        frequency=frequency,
                                        start=start,
                                        end=end)

        return clean_api_data(data, columns_col=columns_col, sum_value_totals=sum_values)

    async def execute_request_async(self, route: str, facet_key: str, columns_col: str = "area-name", sum_values: bool = False,
                                    frequency: str = "monthly", start: str = None, end: str = None, max_concurrent: int = 5,
                                    normalize: bool = True) -> object:
        """
        Async version of execute_request for improved performance
        
        Args:
            route: EIA API route
            facet_key: Key for FACET_PARAMS dictionary
            columns_col: Column(s) to use for pivot columns
            sum_values: Whether to sum values for totals
            frequency: Data frequency (monthly, weekly, etc.)
            start: Start date in YYYY-MM format
            end: End date in YYYY-MM format
            max_concurrent: Maximum concurrent requests
            
        Returns:
            Cleaned DataFrame with requested data
        """
        if not start:
            start = date(2000,1,1).strftime('%Y-%m-%d')
            end = date.today().strftime('%Y-%m-%d')
        if frequency == 'monthly':
            start, end = start[:-3], end[:-3]
        elif frequency == 'annual':
            start, end = start[:4], end[:4]

        async with self.async_client as client:
            data = await client.get_all_data_async(
                route,
                facets=self.FACET_PARAMS[facet_key],
                frequency=frequency,
                start=start,
                end=end
            )

        return clean_api_data(data, columns_col=columns_col, sum_value_totals=sum_values, flatten_columns=False,
                              reset_index=False, normalize_to_bbl=normalize)


    def get_spot_prices(self, start=None, end=None):
        import asyncio as aio
        data = aio.run(self.execute_request_async('/pri/spt',
                                                  facet_key='spot_prices',
                                                  columns_col="product",
                                                  frequency='daily',
                                                  start=start,
                                                  end=end,
                                                  normalize=False))
        return data

    async def get_spot_prices_async(self, start=None, end=None):
        return await self.execute_request_async('/pri/spt',
                                                  facet_key='spot_prices',
                                                  columns_col="product",
                                                  frequency='daily',
                                                  start=start,
                                                  end=end,
                                                  normalize=False)


    def get_refinery_crude_stocks(self, start=None, end=None):
        if not start:
            start = "2001-01"
        if not end:
            end = "2025-08"
            
        data = self.stocks_client.get_all_data('/ref',
                                               data_columns=['value'],
                                               facets=self.FACET_PARAMS['refinery_stocks'],
                                               start=start,
                                               end=end)

        df = clean_api_data(data)

        return df

    async def get_refinery_crude_stocks_async(self, start=None, end=None, max_concurrent=5):
        """
        Async version of get_refinery_crude_stocks for improved performance.
        
        Args:
            start: Start date in YYYY-MM format
            end: End date in YYYY-MM format
            max_concurrent: Maximum concurrent requests
            
        Returns:
            DataFrame with refinery crude stock data
        """
        async with self.client as client:
            data = await client.get_all_data_async(
                '/stoc/ref',
                facets=self.FACET_PARAMS['refinery_stocks'],
                data_columns=['value'],
                frequency='monthly',
                start=start,
                end=end
            )

        return clean_api_data(data)

    def imports_weekly(self, start=None, end=None):
        df = self.execute_request(
            route='/move/wkly',
            facet_key='imports',
            columns_col=("area-name", "product"),
            frequency='weekly',
            start=start,
            end=end
        )
        return df

    def get_padd_imports(self, padd_code, start=None, end=None):
        if not start:
            start = "2001-01"
        if not end:
            end = "2025-08"
            
        params = {
            'duoarea': [padd_code + '-Z00'],
            'process': ["IM0"],
            'product': ["EPC0"]
        }
        data = self.client.move.get_all_data(
            '/impcp',
            data_columns=['value'],
            facets=params,
            start=start,
            end=end)
        df = clean_api_data(data, columns_col='area-name', flatten_columns=False)

        # Rename value columns to include PADD name for uniqueness
        padd_name = rev_codes.get(padd_code.upper(), padd_code)
        if not df.empty:
            # Rename data columns to be unique and descriptive
            column_mapping = {}
            for col in df.columns:
                if col not in ['period', 'units']:
                    # Create unique columns_col name
                    if col == padd_name or col.startswith(padd_name):
                        column_mapping[col] = f"{padd_name}_Total"
                    else:
                        column_mapping[col] = f"{padd_name}_{col}"

            if column_mapping:
                df = df.rename(columns=column_mapping)

        return df

    def get_padd_exports(self, start=None, end=None):
        df = self.execute_request(route='/move/exp',
                                  facet_key='exports',
                                  columns_col=("area-name", "product"),
                                  frequency='monthly',
                                  start=start,
                                  end=end
                                  )
        return df

    def get_padd_imports_from_top_src(self, padd_codes=["R10", "R20", "R30", "R40", "R50"], start=None,
                                      end=None):
        """
        Get imports to specific PADDs from top exporter countries.
        
        Args:
            padd_codes: List of PADD codes (e.g., ["R20", "R30"] for PADD 2 and 3)
            start: Start date in YYYY-MM format
            end: End date in YYYY-MM format
            
        Returns:
            DataFrame with import data by PADD and source country
        """
        if not start:
            start = "2001-01"
        if not end:
            end = "2025-08"
            
        # Create duoarea combinations for selected PADDs and all top exporters
        all_dfs = []

        for padd_code in padd_codes:
            facet_duoareas = []
            for exporter_code in self.agg_level['top_exporter_codes']:
                facet_duoareas.append(f"{padd_code}-{exporter_code}")

            # Create temporary facet params for this specific request
            import_facets = FacetParams(
                duoarea=facet_duoareas,
                process=["IP0"],  # Import process code
                product=["EPC0"],  # Crude oil
                series=self.FACET_PARAMS['imports_to_padd']['series']  # Use existing series filters
            ).params

            try:
                # Execute the request
                data = self.client.get_all_data('/move/impcp',
                                                data_columns=['value'],
                                                facets=import_facets,
                                                frequency='monthly',
                                                start=start,
                                                end=end)

                if len(data) > 0:
                    # Clean and return the data with source country as columns, flatten for HDF5
                    df = clean_api_data(data, columns_col='area-name', flatten_columns=False, reset_index=False)

                    # Rename columns to include PADD name for uniqueness
                    padd_name = rev_codes[padd_code.upper()]
                    column_mapping = {}

                    for col in df.columns:
                        if col not in ['period', 'units']:
                            # Create unique columns_col names with PADD prefix
                            if col == padd_name:
                                column_mapping[col] = (padd_name, "Total")
                            else:
                                column_mapping[col] = (padd_name, col)

                    if column_mapping:
                        df = df.rename(columns=column_mapping)

                    all_dfs.append(df)

                else:
                    # No data from detailed request, try fallback
                    raise ValueError("No detailed data available")

            except Exception as e:
                # Fallback method for PADDs with no detailed source data
                padd_name = rev_codes[padd_code.upper()]
                print(f'Failed to retrieve detailed import data for {padd_name}. Trying fallback method.')
                try:
                    df = self.get_padd_imports(padd_code.upper(), start, end)
                    if df is not None and not df.empty:
                        print("Fallback method was successful")
                        all_dfs.append(df)
                    else:
                        print(f"No data available for {padd_name}")

                except Exception as fallback_error:
                    print(f'Fallback method failed for {padd_name}. Error: {fallback_error}')

        if all_dfs:
            combined_padd_df = pd.concat(all_dfs, axis=1)
            return combined_padd_df
        else:
            print("No data retrieved for any PADD")
            return pd.DataFrame()

    def exports_weekly(self, start, end):
        df = self.execute_request(
            route='/move/wkly',
            facet_key='exports',
            columns_col="product",
            frequency='weekly',
            start=start,
            end=end
        )
        return df

    async def get_exports_async(self, start="2001-01", end="2025-08", max_concurrent=5):
        """
        Async version of get_exports using /move/expcp route 
        
        Args:
            start: Start date in YYYY-MM format
            end: End date in YYYY-MM format
            max_concurrent: Maximum concurrent requests
            
        Returns:
            DataFrame with MultiIndex columns (area-name, product) for export data
        """
        # Use existing execute_request_async with exports facet params
        return await self.execute_request_async(
            route='/move/exp',
            facet_key='exports',
            columns_col=('area-name', 'product'),
            frequency='monthly',
            start=start,
            end=end,
            max_concurrent=max_concurrent
        )

    async def get_imports_async(self, start="2001-01", end="2025-08", max_concurrent=5):

        return await self.execute_request_async(
            route='/move/imp',
            facet_key='imports',
            columns_col=("area-name", "product"),
            frequency="monthly",
            start=start,
            end=end,
            max_concurrent=max_concurrent
        )

    def get_product_stocks(self, start="2001-01", end="2025-08"):

        data = self.client.get_all_data('/sum/snd',
                                        data_columns=['value'],
                                        facets=self.FACET_PARAMS['product_stocks'],
                                        start=start,
                                        end=end)
        df = clean_api_data(data, columns_col=('area-name', 'product'))

        return df

    async def get_product_stocks_async(self, start="2001-01", end="2025-08", max_concurrent=5):
        """
        Async version of get_product_stocks for improved performance.
        
        Args:
            start: Start date in YYYY-MM format
            end: End date in YYYY-MM format
            max_concurrent: Maximum concurrent requests
            
        Returns:
            DataFrame with MultiIndex columns (area-name, product) for product stock data
        """
        # Use existing execute_request_async with product_stocks facet params
        return await self.execute_request_async(
            route='/sum/snd',
            facet_key='product_stocks',
            columns_col=('area-name', 'product'),
            frequency='monthly',
            start=start,
            end=end,
            max_concurrent=max_concurrent
        )

    async def get_product_stocks_weekly(self, start="2001-01-01", end=None):
        if not end:
            end = datetime.today().strftime('%Y-%m-%d')

        return await self.execute_request_async(route='/stoc/wstk',
                                                facet_key='product_stocks',
                                                frequency='weekly',
                                                start=start,
                                                end=end,
                                                columns_col=('duoarea','product'))

    def get_product_supplied(self, start="2001-01-01", end="2025-08-15"):
        data = self.consumption_client.get_all_data('/cons/psup',
                                                    data_columns=['value'],
                                                    facets=self.FACET_PARAMS['product_supplied'],
                                                    frequency="weekly",
                                                    start=start,
                                                    end=end, )

        df = clean_api_data(data, columns_col=("product", 'duoarea'))

        return df

    async def get_product_supplied_async(self, start="2001-01-01", end="2025-08-15", max_concurrent=5):
        """
        Async version of get_product_supplied for improved performance.
        
        Args:
            start: Start date in YYYY-MM-DD format
            end: End date in YYYY-MM-DD format
            max_concurrent: Maximum concurrent requests
            
        Returns:
            DataFrame with product supplied data
        """
        return await self.execute_request_async(
            route='/cons/psup',
            facet_key='product_supplied',
            columns_col=('product', 'duoarea'),
            frequency='weekly',
            start=start,
            end=end,
            max_concurrent=max_concurrent
        )

    def get_tank_farm_stocks(self, start="2001-01", end="2025-08"):
        data = self.stocks_client.get_all_data('/cu',
                                               data_columns=["value"],
                                               facets=self.FACET_PARAMS["tank_farm_stocks"], start=start, end=end)

        df = clean_api_data(data, sum_value_totals=True)

        return df

    async def get_tank_farm_stocks_async(self, start="2001-01", end="2025-08", max_concurrent=5):
        """
        Async version of get_tank_farm_stocks for improved performance.
        
        Args:
            start: Start date in YYYY-MM format
            end: End date in YYYY-MM format
            max_concurrent: Maximum concurrent requests
            
        Returns:
            DataFrame with tank farm stock data
        """


        return await self.execute_request_async(
            '/stoc/cu',
            facet_key='tank_farm_stocks',
            columns_col="area-name",
            frequency='monthly',
            start=start,
            end=end,
            max_concurrent=max_concurrent
        )

    def us_weekly_consumption_breakdown(self, start, end, by_state=True, return_compact=True):

        if by_state:
            facets = copy(self.FACET_PARAMS['consumption_breakdown'])
            facets.update({'duoarea': self.agg_level['PADD'] + ["NUS"]})
        else:
            facets = self.FACET_PARAMS['consumption_breakdown']

        data = self.consumption_client.get_all_data('/wpsup',
                                                    data_columns=["value"],
                                                    facets=facets,
                                                    start=start,
                                                    end=end
                                                    )
        if by_state:
            columns = ('product', 'area-name')
            sum_data = True
        else:
            columns = 'product'
            sum_data = False

        clean_df = clean_api_data(data, date_col='period', columns_col=columns, value_col="value",
                                  sum_value_totals=sum_data)

        return clean_df

    def get_production_by_area(self, start="2001-01", end="2025-08"):

        data = self.client.get_all_data('/crd/crpdn',
                                        data_columns=['value'],
                                        facets=self.FACET_PARAMS['crude_production'],
                                        start=start,
                                        end=end)
        cleaned_prod_df = clean_api_data(data)

        return cleaned_prod_df

    async def get_production_by_area_async(self, start="2001-01", end="2025-08", max_concurrent=5):
        """
        Async version of get_production_by_area for improved performance.
        
        Args:
            start: Start date in YYYY-MM format
            end: End date in YYYY-MM format
            max_concurrent: Maximum concurrent requests
            
        Returns:
            DataFrame with crude oil production data by area
        """
        return await self.execute_request_async(
            route='/crd/crpdn',
            facet_key='crude_production',
            columns_col='area-name',
            frequency='monthly',
            start=start,
            end=end,
            max_concurrent=max_concurrent
        )

    def get_refined_products_production(self, start='2001-01', end="2025-08"):
        data = self.client.get_all_data('/pnp/wprodrb',
                                        data_columns=['value'],
                                        facets=self.FACET_PARAMS['refinery_production'],
                                        start=start,
                                        end=end)

        refinery_df = clean_api_data(data, columns_col=('area-name', 'product'), sum_value_totals=True)

        return refinery_df

    def get_refinery_utilization(self, start="2001-01", end="2025-08"):
        data = self.client.get_all_data('/pnp/wiup',
                                        data_columns=['value'],
                                        facets=self.FACET_PARAMS['refinery_utilization'],
                                        start=start,
                                        end=end)
        refinery_utilization_df = clean_api_data(data, columns_col=('process-name', 'area-name'))

        return refinery_utilization_df

    def get_refinery_consumption(self, start="2001-01", end="2025-08"):
        data = self.client.get_all_data('/pnp/wiup',
                                        data_columns=['value'],
                                        facets=self.FACET_PARAMS['refinery_consumption'],
                                        frequency="weekly",
                                        start=start,
                                        end=end)

        clean_df = clean_api_data(data, columns_col="area-name", sum_value_totals=True)

        return clean_df

    # Async Methods for High-Performance Operations
    async def get_multiple_padd_imports_async(self, padd_codes=None, start="2001-01", end="2025-08",
                                              max_concurrent=5, progress_callback=None):
        """
        Get imports for multiple PADDs concurrently using async client
        
        Args:
            padd_codes: List of PADD codes (e.g., ["R10", "R20", "R30"])
            start: Start date in YYYY-MM format
            end: End date in YYYY-MM format  
            max_concurrent: Maximum concurrent requests
            progress_callback: Optional progress callback function
            
        Returns:
            Combined DataFrame with all PADD import data
        """
        if not padd_codes:
            padd_codes = self.agg_level["PADD"]
        # Create request configurations for each PADD
        requests_config = []
        for padd_code in padd_codes:
            # Create duoarea combinations for this PADD and all exporters
            facet_duoareas = []
            for exporter_code in self.agg_level['top_exporter_codes']:
                facet_duoareas.append(f"{padd_code}-{exporter_code}")

            # Build facet parameters using FacetParams like sync version
            import_facets = FacetParams(
                duoarea=facet_duoareas,
                process=["IP0", "IM0"],  # Import process code
                product=["EPC0"],  # Crude oil
                series=self.FACET_PARAMS['imports_to_padd']['series']  # Use existing series filters
            ).params

            # Structure parameters for async client (like sync version)
            async_params = {
                'facets': import_facets,
                'data_columns': ['value'],
                'frequency': 'monthly',
                'start': start,
                'end': end
            }

            requests_config.append({
                'route': '/petroleum/move/impcp',
                'params': async_params,
                'name': f"PADD_{padd_code}",
                'padd_code': padd_code
            })

        # Execute async requests
        async with AsyncEIAClient(max_concurrent=max_concurrent) as client:
            results = await client.get_multiple_series_async(
                requests_config,
                progress_callback=progress_callback
            )

        # Process results into DataFrames
        all_dfs = []
        for result in results:
            if result['success'] and result['data']:
                padd_code = result['name'].split('_')[1]  # Extract from name
                padd_name = rev_codes[padd_code.upper()]

                # Clean the data with flattened columns for HDF5 compatibility
                df = clean_api_data(result['data'], columns_col='area-name', flatten_columns=True, reset_index=False)

                # Rename columns to include PADD name for uniqueness
                column_mapping = {}
                for col in df.columns:
                    if col not in ['period', 'units']:
                        # Create unique columns_col names with PADD prefix
                        if col == padd_name:
                            column_mapping[col] = (padd_name, "Total")
                        else:
                            column_mapping[col] = (padd_name, col)

                if column_mapping:
                    df = df.rename(columns=column_mapping)
                all_dfs.append(df)
            else:
                print(f"Failed to get data for {result['name']}: {result.get('error', 'Unknown error')}")

        # Combine all DataFrames
        if all_dfs:
            combined_df = pd.concat(all_dfs, axis=1)

            # Remove duplicate columns (like 'period', 'units') that appear in multiple DataFrames
            if len(combined_df.columns) != len(set(combined_df.columns)):
                # Keep only the first occurrence of duplicate columns
                combined_df = combined_df.loc[:, ~combined_df.columns.duplicated()]
                print(f"Removed duplicate columns during concat. Final shape: {combined_df.shape}")

            return combined_df
        else:
            return pd.DataFrame()

    async def bulk_update_petroleum_data_async(self, data_types: List[str] = None, start="2020-01", end="2025-08",
                                               max_concurrent=8, progress_callback=None):
        """
        Bulk update multiple petroleum data types concurrently
        
        Args:
            data_types: List of data types to update (default: all major types)
            start: Start date
            end: End date
            max_concurrent: Maximum concurrent requests
            progress_callback: Progress callback function
            
        Returns:
            Dictionary with results for each data type
        """

        if data_types is None:
            data_types = [
                'refinery_stocks', 'refined_stocks', 'tank_farm_stocks',
                'refinery_production', 'crude_production', 'imports', 'exports'
            ]

        # Create request configurations
        requests_config = []
        route_mapping = {
            'refinery_stocks': '/petroleum/stoc/ref',
            'refined_stocks': '/petroleum/stoc/ts',
            'tank_farm_stocks': '/petroleum/stoc/cu',
            'refinery_production': '/petroleum/pnp/wprodrb',
            'crude_production': '/petroleum/crd/crpdn',
            'imports': '/petroleum/move/wkly',
            'exports': '/petroleum/move/exp'
        }

        for data_type in data_types:
            if data_type in self.FACET_PARAMS and data_type in route_mapping:
                requests_config.append({
                    'route': route_mapping[data_type],
                    'params': {
                        'facets': self.FACET_PARAMS[data_type],
                        'data_columns': ['value'],
                        'frequency': 'monthly',
                        'start': start,
                        'end': end
                    },
                    'name': data_type
                })

        # Execute async requests
        async with AsyncEIAClient(max_concurrent=max_concurrent) as client:
            results = await client.get_multiple_series_async(
                requests_config,
                progress_callback=progress_callback
            )

        # Process and return results
        processed_results = {}
        for result in results:
            data_type = result['name']
            if result['success']:
                # Determine appropriate columns_col structure for each data type
                if data_type in ['refinery_production', 'refined_stocks', 'imports', 'exports']:
                    columns_col = ('area-name', 'product')
                elif data_type == 'tank_farm_stocks':
                    columns_col = 'area-name'
                else:
                    columns_col = 'area-name'

                # Clean the data with flattened columns for HDF5 compatibility
                df = clean_api_data(
                    result['data'],
                    columns_col=columns_col,
                    sum_value_totals=(data_type in ['refinery_production', 'refined_stocks']),
                    flatten_columns=False
                )
                processed_results[data_type] = {
                    'data': df,
                    'success': True,
                    'records': len(result['data'])
                }
            else:
                processed_results[data_type] = {
                    'data': pd.DataFrame(),
                    'success': False,
                    'error': result['error']
                }

        return processed_results

    def get_crude_movements(self, start="2001-01", end="2025-08"):
        """
        Get crude oil movements between PADDs (inter-PADD movements).
        
        Args:
            start: Start date in YYYY-MM format
            end: End date in YYYY-MM format
            
        Returns:
            DataFrame with MultiIndex columns (destination_padd, source_padd) for movement data
        """
        # Create duoarea combinations for all PADD-to-PADD movements
        all_dfs = []

        for dest_padd in self.agg_level['PADD']:
            facet_duoareas = []
            # Create combinations for this destination PADD from all other PADDs
            for src_padd in self.agg_level['PADD']:
                if dest_padd != src_padd:  # No self-movements
                    facet_duoareas.append(f"{dest_padd}-{src_padd}")

            if not facet_duoareas:
                continue

            # Use crude_movements facet parameters with custom duoarea for this destination PADD
            movement_facets = self.FACET_PARAMS['crude_movements'].copy()
            movement_facets['duoarea'] = facet_duoareas

            try:
                # Execute the request
                data = self.client.get_all_data('/petroleum/move/ptb',
                                                data_columns=['value'],
                                                facets=movement_facets,
                                                frequency='monthly',
                                                start=start,
                                                end=end)

                if len(data) > 0:
                    # Clean the data with area-name as columns
                    df = clean_api_data(data, columns_col='area-name', flatten_columns=False, reset_index=False)

                    # Create MultiIndex columns: (destination_padd, source_padd)
                    dest_name = rev_codes[dest_padd.upper()]
                    column_mapping = {}

                    for col in df.columns:
                        if col not in ['period', 'units']:
                            # Parse the area name to determine source PADD
                            # Area names should be in format like "PADD 3" or similar
                            if col in rev_codes.values():
                                # This is a PADD name, create MultiIndex tuple
                                column_mapping[col] = (dest_name, col)
                            elif any(padd_name in col for padd_name in rev_codes.values()):
                                # Extract PADD name from columns_col
                                for padd_code, padd_name in rev_codes.items():
                                    if padd_name in col:
                                        column_mapping[col] = (dest_name, padd_name)
                                        break
                            else:
                                # Fallback - assume it's a source area
                                column_mapping[col] = (dest_name, col)

                    if column_mapping:
                        df = df.rename(columns=column_mapping)

                    all_dfs.append(df)

            except Exception as e:
                dest_name = rev_codes[dest_padd.upper()]
                print(f'Failed to retrieve crude movement data for {dest_name}: {e}')

        if all_dfs:
            combined_df = pd.concat(all_dfs, axis=1)

            # Remove duplicate columns (like 'period', 'units') that appear in multiple DataFrames
            if len(combined_df.columns) != len(set(combined_df.columns)):
                combined_df = combined_df.loc[:, ~combined_df.columns.duplicated()]
                print(f"Removed duplicate columns during concat. Final shape: {combined_df.shape}")

            return combined_df
        else:
            print("No crude movement data retrieved")
            return pd.DataFrame()

    async def get_crude_movements_async(self, start="2001-01", end="2025-08", max_concurrent=5, progress_callback=None):
        """
        Async version of get_crude_movements for inter-PADD crude oil movements.
        
        Args:
            start: Start date in YYYY-MM format
            end: End date in YYYY-MM format
            max_concurrent: Maximum concurrent requests
            progress_callback: Optional progress callback function
            
        Returns:
            DataFrame with MultiIndex columns (destination_padd, source_padd) for movement data
        """

        # Create request configurations for each destination PADD
        requests_config = []
        for dest_padd in self.agg_level['PADD']:
            facet_duoareas = []
            # Create combinations for this destination PADD from all other PADDs
            for src_padd in self.agg_level['PADD']:
                if dest_padd != src_padd:  # No self-movements
                    facet_duoareas.append(f"{dest_padd}-{src_padd}")

            if not facet_duoareas:
                continue

            # Use crude_movements facet parameters with custom duoarea for this destination PADD
            movement_facets = self.FACET_PARAMS['crude_movements'].copy()
            movement_facets['duoarea'] = facet_duoareas

            # Structure parameters for async client
            async_params = {
                'facets': movement_facets,
                'data_columns': ['value'],
                'frequency': 'monthly',
                'start': start,
                'end': end
            }

            requests_config.append({
                'route': '/petroleum/move/ptb',
                'params': async_params,
                'name': f"DEST_{dest_padd}",
                'dest_padd': dest_padd
            })

        # Execute async requests
        async with AsyncEIAClient(max_concurrent=max_concurrent) as client:
            results = await client.get_multiple_series_async(
                requests_config,
                progress_callback=progress_callback
            )

        # Process results into DataFrames
        all_dfs = []
        for result in results:
            if result['success'] and result['data']:
                dest_padd = result['name'].split('_')[1]  # Extract from name
                dest_name = rev_codes[dest_padd.upper()]

                # Clean the data with area-name as columns
                df = clean_api_data(result['data'], columns_col='area-name', flatten_columns=False, reset_index=False)

                # Create MultiIndex columns: (destination_padd, source_padd)
                column_mapping = {}
                for col in df.columns:
                    if col not in ['period', 'units']:
                        # Parse the area name to determine source PADD
                        if col in rev_codes.values():
                            # This is a PADD name, create MultiIndex tuple
                            column_mapping[col] = (dest_name, col)
                        elif any(padd_name in col for padd_name in rev_codes.values()):
                            # Extract PADD name from columns_col
                            for padd_code, padd_name in rev_codes.items():
                                if padd_name in col:
                                    column_mapping[col] = (dest_name, padd_name)
                                    break
                        else:
                            # Fallback - assume it's a source area
                            column_mapping[col] = (dest_name, col)

                if column_mapping:
                    df = df.rename(columns=column_mapping)

                all_dfs.append(df)
            else:
                print(f"Failed to get data for {result['name']}: {result.get('error', 'Unknown error')}")

        # Combine all DataFrames
        if all_dfs:
            combined_df = pd.concat(all_dfs, axis=1)

            # Remove duplicate columns (like 'period', 'units') that appear in multiple DataFrames
            if len(combined_df.columns) != len(set(combined_df.columns)):
                combined_df = combined_df.loc[:, ~combined_df.columns.duplicated()]
                print(f"Removed duplicate columns during concat. Final shape: {combined_df.shape}")

            return combined_df
        else:
            return pd.DataFrame()

    def get_refined_product_movements(self, start="2001-01", end="2025-08"):
        """
        Get refined product movements between PADDs using refined_product_movements FACET_PARAMS.
        
        Args:
            start: Start date in YYYY-MM format
            end: End date in YYYY-MM format
            
        Returns:
            DataFrame with MultiIndex columns (area-name, product) for refined product movement data
        """
        # Execute the request using refined_product_movements facet params
        data = self.client.get_all_data('/move/netr',
                                             data_columns=['value'],
                                             facets=self.FACET_PARAMS['refined_product_movements'],
                                             frequency='monthly',
                                             start=start,
                                             end=end)

        # Clean the data with (area-name, product) as columns structure
        df = clean_api_data(data, columns_col=('area-name', 'product'), flatten_columns=False, reset_index=True)

        return df

    async def get_refined_product_movements_async(self, start="2001-01", end="2025-08", max_concurrent=5):
        """
        Async version of get_refined_product_movements for improved performance.
        
        Args:
            start: Start date in YYYY-MM format
            end: End date in YYYY-MM format
            max_concurrent: Maximum concurrent requests
            
        Returns:
            DataFrame with MultiIndex columns (area-name, product) for refined product movement data
        """
        # Use existing execute_request_async with refined_product_movements facet params
        return await self.execute_request_async(
            route='/move/netr',
            facet_key='refined_product_movements',
            columns_col=('area-name', 'product'),
            frequency='monthly',
            start=start,
            end=end,
            max_concurrent=max_concurrent
        )
