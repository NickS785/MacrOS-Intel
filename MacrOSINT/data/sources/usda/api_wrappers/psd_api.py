"""
USDA FAS Advanced Query API Wrapper
A Python wrapper for the USDA Foreign Agricultural Service Advanced Query API
Supports bulk data retrieval for grains and livestock import/export information.

Author: [Your Name]
Date: July 2025
"""

import requests
import pandas as pd
from typing import List, Optional, Dict, Union
from dataclasses import dataclass, asdict
from enum import Enum
import logging
import json
from datetime import datetime
from MacrOSINT.data.sources.usda.nass_utils import clean_grains, clean_livestock


class CommodityCode(Enum):
    """Common commodity codes for grains and livestock"""
    # Grains
    WHEAT = "0410000"
    WHEAT_FLOUR = "0411000"
    CORN = "0440000"
    BARLEY = "0430000"
    OATS = "0452000"
    SORGHUM = "0459200"
    RICE_MILLED = "0422110"
    RICE_ROUGH = "0422210"
    RYE = "0451000"

    # Livestock/Meat
    BEEF_VEAL = "0111000"
    PORK = "0113000"
    BROILER_MEAT = "0112300"
    TURKEY_MEAT = "0112400"
    SHEEP_GOAT_MEAT = "0112000"

    # Soybeans and products
    SOYBEANS = "2222000"
    SOYBEAN_MEAL = "0813200"
    SOYBEAN_OIL = "4243000"

    # Feed
    CORN_GLUTEN_FEED = "0814300"
    CORN_GLUTEN_MEAL = "0814400"

class CommodityGroup(Enum):
    GRAINS = [
        CommodityCode.WHEAT,
        CommodityCode.WHEAT_FLOUR,
        CommodityCode.CORN,
        CommodityCode.BARLEY,
        CommodityCode.OATS,
        CommodityCode.SORGHUM,
        CommodityCode.RICE_MILLED,
        CommodityCode.RICE_ROUGH,
        CommodityCode.RYE,
    ]
    LIVESTOCK = [
        CommodityCode.BEEF_VEAL,
        CommodityCode.PORK,
        CommodityCode.BROILER_MEAT,
        CommodityCode.TURKEY_MEAT,
        CommodityCode.SHEEP_GOAT_MEAT,
    ]
    SOY_PRODUCTS = [
        CommodityCode.SOYBEANS,
        CommodityCode.SOYBEAN_MEAL,
        CommodityCode.SOYBEAN_OIL,
    ]
    FEED = [
        CommodityCode.CORN_GLUTEN_FEED,
        CommodityCode.CORN_GLUTEN_MEAL,
    ]

REVERSE_GROUP_LOOKUP: Dict[str, CommodityGroup] = {}

for group in CommodityGroup:
    for code_enum in group.value:
        REVERSE_GROUP_LOOKUP[code_enum.value] = group

class AttributeCode(Enum):
    """Common PSD attribute codes"""
    # Production attributes
    AREA_HARVESTED = 4
    BEGINNING_STOCKS = 20
    PRODUCTION = 28
    IMPORTS = 57
    TOTAL_SUPPLY = 58

    # Consumption attributes
    DOMESTIC_CONSUMPTION = 125
    FEED_DOMESTIC_CONSUMPTION = 50
    FOOD_USE_DOMESTIC_CONSUMPTION = 51

    # Trade attributes
    EXPORTS = 88
    ENDING_STOCKS = 176
    TOTAL_DISTRIBUTION = 178

    # Yield/Price attributes
    YIELD = 83

    # Livestock specific
    ANIMAL_NUMBERS = 1
    SLAUGHTER = 29
    CARCASS_WEIGHT = 31
    TOTAL_EXPORTS = 32
    TOTAL_IMPORTS = 36


class CountryCode(Enum):
    """Common country codes"""
    WORLD = "XX"
    UNITED_STATES = "US"
    CHINA = "CH"
    BRAZIL = "BR"
    ARGENTINA = "AR"
    EUROPEAN_UNION = "E2"
    INDIA = "IN"
    RUSSIA = "RS"
    UKRAINE = "UP"
    CANADA = "CA"
    AUSTRALIA = "AS"
    MEXICO = "MX"
    JAPAN = "JA"
    SOUTH_KOREA = "KS"


@dataclass
class QueryParameters:
    """Parameters for USDA FAS Advanced Query"""
    commodities: List[str]
    attributes: List[int]
    countries: List[str]
    marketYears: List[int]

    # Optional parameters with defaults
    queryId: int = 0
    commodityGroupCode: Optional[str] = None
    chkCommoditySummary: bool = False
    chkAttribSummary: bool = False
    chkCountrySummary: bool = False
    commoditySummaryText: str = ""
    attribSummaryText: str = ""
    countrySummaryText: str = ""
    optionColumn: str = "year"
    chkTopCountry: bool = False
    topCountryCount: str = ""
    chkfileFormat: bool = False
    chkPrevMonth: bool = False
    chkMonthChange: bool = False
    chkCodes: bool = False
    chkYearChange: bool = False
    queryName: str = ""
    sortOrder: str = "Commodity/Attribute/Country"
    topCountryState: bool = False



code_lookup = {k.value:k for k in CommodityCode}
comms_dict = dict(cattle=CommodityCode.BEEF_VEAL.value, hogs=CommodityCode.PORK.value, corn = CommodityCode.CORN.value, soybeans = CommodityCode.SOYBEANS.value, soy_meal = CommodityCode.SOYBEAN_MEAL.value)
valid_codes = {v.value for v in CommodityCode}
rev_lookup = {v:k for k,v in comms_dict.items()}
attrs_dict = dict(imports = AttributeCode.IMPORTS, exports = AttributeCode.EXPORTS, consumption = AttributeCode.DOMESTIC_CONSUMPTION, ending_stocks=AttributeCode.ENDING_STOCKS)

# Helper function
def get_commodity_group(code: CommodityCode) -> CommodityGroup:
    for group in CommodityGroup:
        if code in group.value:
            return group
    raise ValueError(f"Commodity code {code} not found in any group.")

class PSD_API:
    """
    Advanced query wrapper for USDA Foreign Agricultural Service API.
    Supports bulk data retrieval with multiple commodities, attributes, and countries.
    """

    def __init__(self, timeout: int = 60):
        """
        Initialize the USDA FAS Advanced API wrapper.

        Args:
            timeout: Request timeout in seconds (default 60 for large queries)
        """
        self.base_url = "https://apps.fas.usda.gov/PSDOnlineApi/api"
        self.timeout = timeout

        # Set up logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)

        # Headers for API requests
        self.headers = {
            "Content-Type": "application/json",
            "Accept": "application/json"
        }
        self.cleaning_func = {
            CommodityGroup.LIVESTOCK:clean_livestock,
            CommodityGroup.GRAINS:clean_grains,
            CommodityGroup.SOY_PRODUCTS:clean_grains
        }

    def _make_request(self, endpoint: str, payload: Dict) -> Union[Dict, List]:
        """
        Make a POST request to the API.

        Args:
            endpoint: API endpoint path
            payload: Request payload dictionary

        Returns:
            Response data (dict or list)

        Raises:
            requests.exceptions.RequestException: If request fails
        """
        url = f"{self.base_url}{endpoint}"
        self.logger.info(f"Making request to: {url}")
        self.logger.debug(f"Payload: {json.dumps(payload, indent=2)}")

        response = requests.post(
            url,
            headers=self.headers,
            json=payload,
            timeout=self.timeout
        )

        response.raise_for_status()
        return response.json()

    def run_query(self, params: QueryParameters) -> pd.DataFrame:
        """
        Execute an advanced query with the specified parameters.

        Args:
            params: QueryParameters object with query specifications

        Returns:
            DataFrame with query results
        """
        # Convert dataclass to dict and remove None values
        payload = {k: v for k, v in asdict(params).items() if v is not None and k is not "group"}
        years = payload['marketYears']
        code = payload['commodities']
        comm_group = REVERSE_GROUP_LOOKUP[code[0]]

        if isinstance(comm_group, List) and len(pd.Series(comm_group).unique()) < 2:
            comm_group = comm_group[0]

        try:
            data = self._make_request("/query/RunQuery", payload)

            # Handle different response formats
            if isinstance(data, list):
                df = pd.DataFrame(data)
            elif isinstance(data, dict):
                # If the response is wrapped in a results key
                if 'queryResult' in data:
                    df = pd.DataFrame(data['queryResult'])
                else:
                    df = pd.DataFrame([data])
            else:
                raise ValueError(f"Unexpected response format: {type(data)}")

            # Add metadata columns if not present
            if not df.empty:
                df['query_date'] = datetime.now()


            df = self.cleaning_func[comm_group](df)

            return df

        except Exception as e:
            self.logger.error(f"Query failed: {e}")
            raise

    def get_grain_trade_data(
            self,
            grains: Optional[List[Union[str, CommodityCode]]] = None,
            countries: Optional[List[Union[str, CountryCode]]] = None,
            years: Optional[List[int]] = None,
            include_production: bool = True,
            include_trade: bool = True
    ) -> pd.DataFrame:
        """
        Get comprehensive grain trade data.

        Args:
            grains: List of grain commodities (defaults to major grains)
            countries: List of countries (defaults to major producers/consumers)
            years: List of marketing years (defaults to last 5 years)
            include_production: Include production/supply attributes
            include_trade: Include import/export attributes

        Returns:
            DataFrame with grain trade data
        """
        # Default grains if not specified
        if grains is None:
            grains = [
                CommodityCode.WHEAT,
                CommodityCode.CORN,
                CommodityCode.RICE_MILLED,
                CommodityCode.BARLEY,
                CommodityCode.SORGHUM
            ]

        # Default countries if not specified
        if countries is None:
            countries = [
                CountryCode.UNITED_STATES,
                CountryCode.CHINA,
                CountryCode.BRAZIL,
                CountryCode.ARGENTINA,
                CountryCode.EUROPEAN_UNION,
                CountryCode.INDIA,
                CountryCode.RUSSIA,
                CountryCode.UKRAINE
            ]

        # Default to last 5 years if not specified
        if years is None:
            current_year = datetime.now().year
            years = list(range(current_year - 4, current_year + 1))

        # Build attribute list
        attributes = []
        if include_production:
            attributes.extend([
                AttributeCode.AREA_HARVESTED.value,
                AttributeCode.PRODUCTION.value,
                AttributeCode.YIELD.value,
                AttributeCode.BEGINNING_STOCKS.value
            ])
        if include_trade:
            attributes.extend([
                AttributeCode.IMPORTS.value,
                AttributeCode.EXPORTS.value,
                AttributeCode.ENDING_STOCKS.value,
                AttributeCode.DOMESTIC_CONSUMPTION.value
            ])

        # Convert enums to values
        commodity_codes = [c.value if isinstance(c, CommodityCode) else c for c in grains]
        country_codes = [c.value if isinstance(c, CountryCode) else c for c in countries]

        params = QueryParameters(
            commodities=commodity_codes,
            attributes=attributes,
            countries=country_codes,
            marketYears=years
        )

        return self.run_query(params)

    def get_livestock_trade_data(
            self,
            meats: Optional[List[Union[str, CommodityCode]]] = None,
            countries: Optional[List[Union[str, CountryCode]]] = None,
            years: Optional[List[int]] = None
    ) -> pd.DataFrame:
        """
        Get comprehensive livestock/meat trade data.

        Args:
            meats: List of meat commodities (defaults to major meats)
            countries: List of countries (defaults to major producers/consumers)
            years: List of marketing years (defaults to last 5 years)

        Returns:
            DataFrame with livestock trade data
        """
        # Default meats if not specified
        if meats is None:
            meats = [
                CommodityCode.BEEF_VEAL,
                CommodityCode.PORK,
                CommodityCode.BROILER_MEAT
            ]

        # Default countries if not specified
        if countries is None:
            countries = [
                CountryCode.UNITED_STATES,
                CountryCode.BRAZIL,
                CountryCode.CHINA,
                CountryCode.EUROPEAN_UNION,
                CountryCode.AUSTRALIA,
                CountryCode.INDIA,
                CountryCode.ARGENTINA,
                CountryCode.CANADA
            ]

        # Default to last 5 years if not specified
        if years is None:
            current_year = datetime.now().year
            years = list(range(current_year - 4, current_year + 1))

        # Livestock-specific attributes
        attributes = [
            AttributeCode.PRODUCTION.value,
            AttributeCode.IMPORTS.value,
            AttributeCode.EXPORTS.value,
            AttributeCode.DOMESTIC_CONSUMPTION.value,
            AttributeCode.BEGINNING_STOCKS.value,
            AttributeCode.ENDING_STOCKS.value
        ]

        # Convert enums to values
        commodity_codes = [c.value if isinstance(c, CommodityCode) else c for c in meats]
        country_codes = [c.value if isinstance(c, CountryCode) else c for c in countries]

        params = QueryParameters(
            commodities=commodity_codes,
            attributes=attributes,
            countries=country_codes,
            marketYears=years
        )


        return self.run_query(params)

    def get_trade_balance(
            self,
            commodities: List[Union[str, CommodityCode]],
            countries: List[Union[str, CountryCode]],
            years: List[int]
    ) -> pd.DataFrame:
        """
        Calculate trade balance (exports - imports) for specified commodities.

        Args:
            commodities: List of commodity codes
            countries: List of country codes
            years: List of marketing years

        Returns:
            DataFrame with trade balance calculations
        """
        # Get only import/export data
        attributes = [
            AttributeCode.IMPORTS.value,
            AttributeCode.EXPORTS.value
        ]

        # Convert enums to values
        commodity_codes = [c.value if isinstance(c, CommodityCode) else c for c in commodities]
        country_codes = [c.value if isinstance(c, CountryCode) else c for c in countries]

        params = QueryParameters(
            commodities=commodity_codes,
            attributes=attributes,
            countries=country_codes,
            marketYears=years
        )

        df = self.run_query(params)

        # Calculate trade balance if data is available
        if not df.empty and 'attributeId' in df.columns:
            # Pivot to get imports and exports in separate columns
            pivot_df = df.pivot_table(
                index=['countryCode', 'commodityCode', 'marketYear'],
                columns='attributeId',
                values='value',
                aggfunc='first'
            ).reset_index()

            # Calculate trade balance
            if AttributeCode.IMPORTS.value in pivot_df.columns and AttributeCode.EXPORTS.value in pivot_df.columns:
                pivot_df['trade_balance'] = (
                        pivot_df[AttributeCode.EXPORTS.value] - pivot_df[AttributeCode.IMPORTS.value]
                )
                pivot_df['net_exporter'] = pivot_df['trade_balance'] > 0

            return pivot_df

        return df

    def get_supply_demand_summary(
            self,
            commodity: Union[str, CommodityCode],
            country: Union[str, CountryCode],
            years: Optional[List[int]] = None
    ) -> pd.DataFrame:
        """
        Get complete supply and demand balance sheet for a commodity/country.

        Args:
            commodity: Commodity code
            country: Country code
            years: List of marketing years (defaults to last 10 years)

        Returns:
            DataFrame with supply/demand balance
        """
        if years is None:
            current_year = datetime.now().year
            years = list(range(current_year - 9, current_year + 1))

        # Get all relevant attributes for S&D balance
        attributes = [
            AttributeCode.BEGINNING_STOCKS.value,
            AttributeCode.PRODUCTION.value,
            AttributeCode.IMPORTS.value,
            AttributeCode.TOTAL_SUPPLY.value,
            AttributeCode.DOMESTIC_CONSUMPTION.value,
            AttributeCode.EXPORTS.value,
            AttributeCode.ENDING_STOCKS.value,
            AttributeCode.TOTAL_DISTRIBUTION.value
        ]

        commodity_code = commodity.value if isinstance(commodity, CommodityCode) else commodity
        country_code = country.value if isinstance(country, CountryCode) else country

        params = QueryParameters(
            commodities=[commodity_code],
            attributes=attributes,
            countries=[country_code],
            marketYears=years
        )

        df = self.run_query(params)


        # Pivot for easier reading
        if not df.empty and 'attributeId' in df.columns:
            pivot_df = df.pivot_table(
                index='marketYear',
                columns='attributeName',
                values='value',
                aggfunc='first'
            )
            return pivot_df

        return df


# Example usage
if __name__ == "__main__":
    # Initialize the API wrapper
    api = PSD_API()

    # Example 1: Get trade data for major grains
    print("Fetching grain trade data...")
    grain_data = api.get_grain_trade_data(
        grains=[CommodityCode.WHEAT, CommodityCode.CORN],
        countries=[CountryCode.UNITED_STATES, CountryCode.CHINA],
        years=[2023, 2024, 2025]
    )
    print(f"Retrieved {len(grain_data)} records")
    print(grain_data.head())

    # Example 2: Get beef trade data
    print("\nFetching beef trade data...")
    beef_data = api.get_livestock_trade_data(
        meats=[CommodityCode.BEEF_VEAL],
        countries=[CountryCode.UNITED_STATES, CountryCode.BRAZIL],
        years=[2024, 2025]
    )
    print(beef_data.head())

    # Example 3: Calculate trade balance
    print("\nCalculating trade balance...")
    trade_balance = api.get_trade_balance(
        commodities=[CommodityCode.WHEAT, CommodityCode.BEEF_VEAL],
        countries=[CountryCode.UNITED_STATES],
        years=[2025]
    )
    print(trade_balance)

    # Example 4: Get supply/demand summary
    print("\nFetching supply/demand summary...")
    supply_demand = api.get_supply_demand_summary(
        commodity=CommodityCode.CORN,
        country=CountryCode.UNITED_STATES,
        years=[2023, 2024, 2025]
    )
    print(supply_demand)

    # Example 5: Custom query with specific parameters
    print("\nRunning custom query...")
    custom_params = QueryParameters(
        commodities=["0111000"],  # Beef
        attributes=[57],  # Imports
        countries=["US"],
        marketYears=list(range(2020, 2026))
    )
    custom_data = api.run_query(custom_params)
    print(custom_data.head())
