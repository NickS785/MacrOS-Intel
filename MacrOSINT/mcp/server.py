"""
MCP Server for MacrOS-Intel EIA Data
Exposes NatGasHelper and PetroleumHelper as MCP tools.
"""
from mcp.server.fastmcp import FastMCP
from typing import Literal, Optional

from MacrOSINT.data.sources.eia.api_tools import (
    NatGasHelper, PetroleumHelper, gas_consumers, states_plus_usa, clean_api_data,
)
from MacrOSINT.mcp.serializers import format_df_for_llm

mcp_server = FastMCP("MacrOS-Intel EIA Data Server")

# ---------------------------------------------------------------------------
# Initialize helpers (singletons for the server lifetime)
# ---------------------------------------------------------------------------
ng_helper = NatGasHelper()
pet_helper = PetroleumHelper()

# ---------------------------------------------------------------------------
# Type literals derived from the actual PARAMS/FACET_PARAMS keys
# ---------------------------------------------------------------------------
NatGasParamKey = Literal[
    "underground_storage",
    "spot_prices",
    "state_pct_of_consumption",
    "consolidated_consumption",
    "production",
    "state_production_detailed",
]

PetroleumFacetKey = Literal[
    "spot_prices",
    "imports",
    "exports",
    "imports_to_padd",
    "exports_from_padd",
    "product_stocks",
    "crude_movements",
    "refined_product_movements",
    "tank_farm_stocks",
    "refinery_stocks",
    "refined_stocks",
    "refinery_production",
    "consumption_breakdown",
    "crude_production",
    "refinery_consumption",
    "product_supplied",
    "refinery_utilization",
    "receipts_from_producers",
]

PetroleumRoute = Literal[
    "/sum/snd",
    "/move/imp",
    "/move/exp",
    "/move/impcp",
    "/move/netr",
    "/move/wkly",
    "/stoc/ref",
    "/stoc/cu",
    "/stoc/wstk",
    "/crd/crpdn",
    "/pnp/wprodrb",
    "/pnp/wiup",
    "/cons/psup",
    "/pri/spt",
]


# ===================================================================
# Discovery Tools
# ===================================================================

@mcp_server.tool()
def list_natgas_datasets() -> str:
    """
    List all available Natural Gas dataset keys that can be fetched.
    Returns the parameter keys from NatGasHelper.PARAMS with their
    route and frequency info.
    """
    lines = []
    for key, cp in ng_helper.PARAMS.items():
        p = cp.params
        lines.append(
            f"- {key}: route={p.get('route')}, "
            f"frequency={p.get('frequency')}, "
            f"columns_col={p.get('columns_col')}"
        )
    return "Available Natural Gas datasets:\n" + "\n".join(lines)


@mcp_server.tool()
def list_petroleum_datasets() -> str:
    """
    List all available Petroleum dataset keys (facet configurations).
    Returns the FACET_PARAMS keys from PetroleumHelper with facet summaries.
    """
    lines = []
    for key, facets in pet_helper.FACET_PARAMS.items():
        products = facets.get("product", [])
        processes = facets.get("process", [])
        lines.append(
            f"- {key}: products={products}, processes={processes}"
        )
    return "Available Petroleum datasets:\n" + "\n".join(lines)


# ===================================================================
# Natural Gas Tools
# ===================================================================

@mcp_server.tool()
def fetch_natgas_data(
    param_key: NatGasParamKey,
    start: Optional[str] = None,
    end: Optional[str] = None,
    max_rows: int = 150,
    summarize: bool = False,
) -> str:
    """
    Fetch Natural Gas data from the EIA API using NatGasHelper.

    Args:
        param_key: Dataset key - one of: underground_storage, spot_prices,
                   state_pct_of_consumption, consolidated_consumption,
                   production, state_production_detailed
        start: Start date (format depends on frequency, e.g. '2020-01' for monthly,
               '2020-01-01' for weekly)
        end: End date in same format as start
        max_rows: Maximum rows to return (default 150, most recent)
        summarize: If True, return statistical summary instead of raw data
    """
    try:
        df = ng_helper.execute_request(param_key=param_key, start=start, end=end)
        return format_df_for_llm(df, max_rows=max_rows, summarize=summarize)
    except Exception as e:
        return f"Error fetching natural gas data for '{param_key}': {e}"


@mcp_server.tool()
async def fetch_natgas_data_async(
    param_key: NatGasParamKey,
    start: Optional[str] = None,
    end: Optional[str] = None,
    max_rows: int = 150,
    summarize: bool = False,
) -> str:
    """
    Async version: Fetch Natural Gas data from the EIA API.

    Args:
        param_key: Dataset key (same options as fetch_natgas_data)
        start: Start date
        end: End date
        max_rows: Maximum rows to return
        summarize: If True, return statistical summary
    """
    try:
        df = await ng_helper.execute_request_async(
            param_key=param_key, start=start, end=end
        )
        return format_df_for_llm(df, max_rows=max_rows, summarize=summarize)
    except Exception as e:
        return f"Error fetching natural gas data for '{param_key}': {e}"


@mcp_server.tool()
def fetch_natgas_consumption(
    areas: Optional[str] = None,
    start: Optional[str] = None,
    end: Optional[str] = None,
    max_rows: int = 100,
    summarize: bool = False,
) -> str:
    """
    Fetch Natural Gas consumption data by region or state.

    Accepts EIA duoarea codes and/or region group names. Values are
    comma-separated. Region names are expanded to their member states.

    Regions:
        East   -> SPA,SNY,SVA,SCT,SNC,SNJ,SMA,SMD
        Midwest-> SOH,SOK,SIN,SIL,SMI,SIA,SMN,SMO
        South  -> SAL,SGA,SLA,SMS,STX,SAR,SSC,SFL
        West   -> SCA,SAZ,SWA,SOR,SND,SCO,SNV,SUT

    State duoarea codes use the EIA convention: 'S' + 2-letter postal code
    (e.g. SPA=Pennsylvania, STX=Texas, SNY=New York). NUS=US total.

    Args:
        areas: Comma-separated region names and/or duoarea codes.
               Examples: 'East', 'East,South', 'SPA,SNY,STX', 'East,SCA'.
               Default (None) returns all regions grouped.
        start: Start date in YYYY-MM format (default: full history)
        end: End date in YYYY-MM format (default: latest available)
        max_rows: Maximum rows to return (default 100, most recent)
        summarize: If True, return statistical summary instead of raw data
    """
    import pandas as pd

    try:
        # Default: all regions grouped (original behavior)
        if areas is None:
            df = ng_helper.regional_consumption_sync(start=start, end=end)
            return format_df_for_llm(df, max_rows=max_rows, summarize=summarize)

        # Parse input tokens
        tokens = [t.strip() for t in areas.split(",") if t.strip()]

        # Separate region names from raw duoarea codes
        region_names = []
        duoarea_codes = []
        for token in tokens:
            if token in gas_consumers:
                region_names.append(token)
            else:
                # Accept as-is (e.g. SPA, NUS, R31)
                duoarea_codes.append(token)

        # Build request via the consolidated_consumption ClientParams
        CParams = ng_helper.PARAMS['consolidated_consumption']
        if start or end:
            CParams._add_start_params(start, end)

        # If only region names and no loose codes, group by region
        if region_names and not duoarea_codes:
            region_dfs = {}
            for region in region_names:
                CParams.update_facets('duoarea', gas_consumers[region])
                req = CParams.request()
                route = req.pop('route')
                data = ng_helper.client.get_all_data(route, **req)
                CParams.update_clean(drop_cols=['units'], reset_index=False)
                region_dfs[region] = clean_api_data(data, **CParams.clean())
            df = pd.concat(region_dfs, axis=1, keys=region_dfs.keys())
            return format_df_for_llm(df, max_rows=max_rows, summarize=summarize)

        # Expand any region names to their member codes
        all_codes = list(duoarea_codes)
        for region in region_names:
            all_codes.extend(gas_consumers[region])
        # Deduplicate while preserving order
        seen = set()
        unique_codes = []
        for c in all_codes:
            if c not in seen:
                seen.add(c)
                unique_codes.append(c)

        CParams.update_facets('duoarea', unique_codes)
        req = CParams.request()
        route = req.pop('route')
        data = ng_helper.client.get_all_data(route, **req)
        CParams.update_clean(drop_cols=['units'], reset_index=True)
        df = clean_api_data(data, **CParams.clean())
        return format_df_for_llm(df, max_rows=max_rows, summarize=summarize)

    except Exception as e:
        return f"Error fetching consumption data for areas='{areas}': {e}"


# ===================================================================
# Petroleum Tools
# ===================================================================

@mcp_server.tool()
def fetch_petroleum_data(
    route: PetroleumRoute,
    facet_key: PetroleumFacetKey,
    columns_col: str = "area-name",
    frequency: str = "monthly",
    start: Optional[str] = None,
    end: Optional[str] = None,
    sum_values: bool = False,
    max_rows: int = 150,
    summarize: bool = False,
) -> str:
    """
    Fetch Petroleum data from the EIA API using PetroleumHelper.

    Args:
        route: EIA API sub-route (e.g. '/sum/snd', '/move/imp', '/crd/crpdn')
        facet_key: FACET_PARAMS key to select the dataset configuration
        columns_col: Column(s) to pivot on. Use 'area-name' for single level,
                     or 'area-name,product' (comma-separated) for multi-level
        frequency: Data frequency - monthly, weekly, daily, or annual
        start: Start date in YYYY-MM format (or YYYY-MM-DD for weekly/daily)
        end: End date
        sum_values: Whether to add total columns
        max_rows: Maximum rows to return
        summarize: If True, return statistical summary
    """
    try:
        # Parse comma-separated columns_col into tuple if needed
        if "," in columns_col:
            col_tuple = tuple(c.strip() for c in columns_col.split(","))
        else:
            col_tuple = columns_col

        df = pet_helper.execute_request(
            route=route,
            facet_key=facet_key,
            columns_col=col_tuple,
            sum_values=sum_values,
            frequency=frequency,
            start=start,
            end=end,
        )
        return format_df_for_llm(df, max_rows=max_rows, summarize=summarize)
    except Exception as e:
        return f"Error fetching petroleum data (route={route}, facet={facet_key}): {e}"


@mcp_server.tool()
def fetch_petroleum_spot_prices(
    start: Optional[str] = None,
    end: Optional[str] = None,
    max_rows: int = 200,
) -> str:
    """
    Fetch petroleum spot prices (WTI, Diesel, Jet Fuel, RBOB Gasoline).
    Returns daily price data from the EIA.

    Args:
        start: Start date YYYY-MM-DD
        end: End date YYYY-MM-DD
        max_rows: Maximum rows to return (default 200)
    """
    try:
        df = pet_helper.get_spot_prices(start=start, end=end)
        return format_df_for_llm(df, max_rows=max_rows)
    except Exception as e:
        return f"Error fetching petroleum spot prices: {e}"


@mcp_server.tool()
def fetch_petroleum_stocks(
    start: str = "2020-01",
    end: Optional[str] = None,
    max_rows: int = 150,
    summarize: bool = False,
) -> str:
    """
    Fetch petroleum product stocks by PADD region and product type.
    Includes crude oil, motor gasoline, distillate, and jet fuel stocks.

    Args:
        start: Start date YYYY-MM
        end: End date YYYY-MM
        max_rows: Maximum rows
        summarize: If True, return summary statistics
    """
    try:
        df = pet_helper.get_product_stocks(start=start, end=end)
        return format_df_for_llm(df, max_rows=max_rows, summarize=summarize)
    except Exception as e:
        return f"Error fetching petroleum stocks: {e}"


@mcp_server.tool()
def fetch_crude_production(
    start: str = "2020-01",
    end: Optional[str] = None,
    max_rows: int = 150,
) -> str:
    """
    Fetch crude oil production by PADD region.

    Args:
        start: Start date YYYY-MM
        end: End date YYYY-MM
        max_rows: Maximum rows
    """
    try:
        df = pet_helper.get_production_by_area(start=start, end=end)
        return format_df_for_llm(df, max_rows=max_rows)
    except Exception as e:
        return f"Error fetching crude production: {e}"


@mcp_server.tool()
def fetch_refinery_utilization(
    start: str = "2020-01",
    end: Optional[str] = None,
    max_rows: int = 150,
) -> str:
    """
    Fetch refinery utilization rates and operating capacity by PADD region.

    Args:
        start: Start date YYYY-MM
        end: End date YYYY-MM
        max_rows: Maximum rows
    """
    try:
        df = pet_helper.get_refinery_utilization(start=start, end=end)
        return format_df_for_llm(df, max_rows=max_rows)
    except Exception as e:
        return f"Error fetching refinery utilization: {e}"


@mcp_server.tool()
def fetch_petroleum_imports(
    padd_codes: str = "R10,R20,R30,R40,R50",
    start: str = "2020-01",
    end: Optional[str] = None,
    max_rows: int = 150,
) -> str:
    """
    Fetch crude oil imports by PADD region from top exporter countries
    (Canada, Saudi Arabia, Iraq, Mexico).

    Args:
        padd_codes: Comma-separated PADD codes (R10=East Coast, R20=Midwest,
                    R30=Gulf Coast, R40=Rocky Mountain, R50=West Coast)
        start: Start date YYYY-MM
        end: End date YYYY-MM
        max_rows: Maximum rows
    """
    try:
        codes = [c.strip() for c in padd_codes.split(",")]
        df = pet_helper.get_padd_imports_from_top_src(
            padd_codes=codes, start=start, end=end
        )
        return format_df_for_llm(df, max_rows=max_rows)
    except Exception as e:
        return f"Error fetching petroleum imports: {e}"


@mcp_server.tool()
def fetch_petroleum_exports(
    start: str = "2020-01",
    end: Optional[str] = None,
    max_rows: int = 150,
) -> str:
    """
    Fetch petroleum exports by PADD region and product type.

    Args:
        start: Start date YYYY-MM
        end: End date YYYY-MM
        max_rows: Maximum rows
    """
    try:
        df = pet_helper.get_padd_exports(start=start, end=end)
        return format_df_for_llm(df, max_rows=max_rows)
    except Exception as e:
        return f"Error fetching petroleum exports: {e}"


@mcp_server.tool()
def fetch_crude_movements(
    start: str = "2020-01",
    end: Optional[str] = None,
    max_rows: int = 150,
) -> str:
    """
    Fetch inter-PADD crude oil pipeline movements.
    Shows flows between PADD regions.

    Args:
        start: Start date YYYY-MM
        end: End date YYYY-MM
        max_rows: Maximum rows
    """
    try:
        df = pet_helper.get_crude_movements(start=start, end=end)
        return format_df_for_llm(df, max_rows=max_rows)
    except Exception as e:
        return f"Error fetching crude movements: {e}"


@mcp_server.tool()
def fetch_refined_product_movements(
    start: str = "2020-01",
    end: Optional[str] = None,
    max_rows: int = 150,
) -> str:
    """
    Fetch refined product (gasoline, distillate, jet fuel) net receipts
    between PADD regions.

    Args:
        start: Start date YYYY-MM
        end: End date YYYY-MM
        max_rows: Maximum rows
    """
    try:
        df = pet_helper.get_refined_product_movements(start=start, end=end)
        return format_df_for_llm(df, max_rows=max_rows)
    except Exception as e:
        return f"Error fetching refined product movements: {e}"


@mcp_server.tool()
def fetch_refinery_production(
    start: str = "2020-01",
    end: Optional[str] = None,
    max_rows: int = 150,
) -> str:
    """
    Fetch refinery production of gasoline, distillate, and jet fuel by PADD.

    Args:
        start: Start date YYYY-MM
        end: End date YYYY-MM
        max_rows: Maximum rows
    """
    try:
        df = pet_helper.get_refined_products_production(start=start, end=end)
        return format_df_for_llm(df, max_rows=max_rows)
    except Exception as e:
        return f"Error fetching refinery production: {e}"


@mcp_server.tool()
def fetch_tank_farm_stocks(
    start: str = "2020-01",
    end: Optional[str] = None,
    max_rows: int = 150,
) -> str:
    """
    Fetch Cushing, OK tank farm crude oil stocks and other PADD storage.

    Args:
        start: Start date YYYY-MM
        end: End date YYYY-MM
        max_rows: Maximum rows
    """
    try:
        df = pet_helper.get_tank_farm_stocks(start=start, end=end)
        return format_df_for_llm(df, max_rows=max_rows)
    except Exception as e:
        return f"Error fetching tank farm stocks: {e}"


# ===================================================================
# Server Entry Point
# ===================================================================

if __name__ == "__main__":
    mcp_server.run()
