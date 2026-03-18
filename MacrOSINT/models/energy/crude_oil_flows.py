import pandas as pd
import numpy as np
from typing import Optional, Iterable, Dict
from MacrOSINT.data.data_tables import PETable
from MacrOSINT.components.plotting.maps import petrol_maps as pm
import plotly.io as pio

pio.renderers.default = "browser"
PADD_CODES = {"PADD1","PADD2","PADD3","PADD4","PADD5"}
COUNTRY_NCODES = { "NCA":'Canada', "NSA":'Saudi Arabia', "NIZ":"Iraq", "NMX": "Mexico"}

def _norm_padd(x: str) -> Optional[str]:
    """Normalize inputs like 'PADD 3', 'padd3', '3' → 'PADD3'."""
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return None
    s = str(x).strip().upper().replace(" ", "")
    if s.startswith("PADD"):
        code = s
    elif s in {"1","2","3","4","5"}:
        code = f"PADD{s}"
    else:
        return None
    return code if code in PADD_CODES else None


def _to_bbl(val, unit: str) -> float:
    """Convert various oil volume units to barrels (float)."""
    if pd.isna(val):
        return np.nan
    v = float(val)
    unit = (unit or "bbl").lower()
    if unit in {"bbl", "barrels"}:
        return v
    if unit in {"kb", "kbbl", "kbbls", "thousand_barrels"}:
        return v * 1_000.0
    if unit in {"mb", "mbbl", "mbbls", "million_barrels"}:
        return v * 1_000_000.0
    if unit in {"bpd", "bbl_per_day"}:
        # caller should pass a period-total; if they pass a rate, convert upstream
        return v
    raise ValueError(f"Unsupported unit: {unit}")

def _greedy_balance_pairs(net_receipts: pd.Series) -> pd.DataFrame:
    """
    Turn net receipts by PADD (positive=inflow, negative=outflow) into
    pairwise flows using a simple greedy algorithm.
    Returns columns: origin_area, dest_area, value_bbl
    """
    pos = net_receipts[net_receipts > 0].sort_values(ascending=False).copy()
    neg = (-net_receipts[net_receipts < 0]).sort_values(ascending=False).copy()

    flows = []
    i_pos = 0
    i_neg = 0
    pos_items = list(pos.items())  # [(padd, vol_in)]
    neg_items = list(neg.items())  # [(padd, vol_out)]

    while i_pos < len(pos_items) and i_neg < len(neg_items):
        dst, need = pos_items[i_pos]
        src, avail = neg_items[i_neg]
        vol = min(need, avail)
        if vol > 0:
            flows.append({"origin_area": src, "dest_area": dst, "value_bbl": float(vol)})
        need -= vol
        avail -= vol
        # advance pointers
        if need <= 1e-6:
            i_pos += 1
        else:
            pos_items[i_pos] = (dst, need)
        if avail <= 1e-6:
            i_neg += 1
        else:
            neg_items[i_neg] = (src, avail)

    return pd.DataFrame(flows)

def _greedy_pairs_from_net(
    net_df: pd.DataFrame, area_col: str, value_col: str, product_col: str
) -> pd.DataFrame:
    """
    For each non-crude product, turn PADD net receipts (pos=inflow, neg=outflow)
    into pairwise flows using a greedy allocator.
    Returns: origin_area, dest_area, product, value_bbl
    """
    out = []
    for prod, grp in net_df.groupby(product_col, dropna=False):
        # Build signed series by area
        s = grp.groupby(area_col)[value_col].sum().dropna()
        s = s[s.index.map(lambda a: a in PADD_CODES)]
        if s.empty:
            continue
        # Nudge to zero sum if tiny residual exists
        if abs(s.sum()) > 1e-6:
            idx = s.abs().idxmax()
            s.loc[idx] = s.loc[idx] - s.sum()
        pos = s[s > 0].sort_values(ascending=False).copy()
        neg = (-s[s < 0]).sort_values(ascending=False).copy()
        pos_items = list(pos.items()); neg_items = list(neg.items())
        i_pos = i_neg = 0
        while i_pos < len(pos_items) and i_neg < len(neg_items):
            dst, need = pos_items[i_pos]
            src, avail = neg_items[i_neg]
            vol = float(min(need, avail))
            if vol > 0:
                out.append({"origin_area": src, "dest_area": dst, "product": prod, "value_bbl": vol})
            need -= vol; avail -= vol
            if need <= 1e-6: i_pos += 1
            else:             pos_items[i_pos] = (dst, need)
            if avail <= 1e-6: i_neg += 1
            else:             neg_items[i_neg] = (src, avail)
    return pd.DataFrame(out)
def _is_crude(product: Optional[str], crude_keys: Iterable[str]) -> bool:
    if product is None or pd.isna(product):
        return False
    p = str(product).strip().lower()
    return any(ck in p for ck in crude_keys)



def build_flow_dataframe(
    period: str,
    *,
    # IMPORTS (crude only)
    imports_df: Optional[pd.DataFrame] = None,
    imports_cols: Dict[str,str] = None,         # {"period","dest_area","value", optional: "origin_country","product"}
    imports_unit: str = "bbl",
    crude_keys: Iterable[str] = ("crude","crude oil","cushing","wticrude","brent"),
    # EXPORTS (crude + refined)
    exports_df: Optional[pd.DataFrame] = None,
    exports_cols: Dict[str,str] = None,         # {"period","origin_area","value", optional: "dest_country","product"}
    exports_unit: str = "bbl",
    # INTER-PADD PAIRS (crude only)
    inter_padd_pairs_df: Optional[pd.DataFrame] = None,
    inter_pairs_cols: Dict[str,str] = None,     # {"period","origin_area","dest_area","value", optional: "product"}
    inter_pairs_unit: str = "bbl",
    # NET RECEIPTS (crude + refined) → use **refined only** to synthesize refined inter-PADD
    net_receipts_df: Optional[pd.DataFrame] = None,
    net_receipts_cols: Dict[str,str] = None,    # {"period","area","value","product"}
    net_receipts_unit: str = "bbl",
    # STOCKS (crude + refined)
    stocks_df: Optional[pd.DataFrame] = None,
    stocks_cols: Dict[str,str] = None,          # {"period","source_area","value","product"}
    stocks_unit: str = "bbl",
) -> pd.DataFrame:
    """
    Output columns:
      direction ('import'|'export'|'inter_padd'|'stocks'),
      origin_area, dest_area, origin_country, dest_country, product, value_bbl, period
    """
    def _get(df, keymap, k):
        return df[keymap[k]] if (df is not None and keymap and k in keymap and keymap[k] in df.columns) else None

    flows = []

    # ---- IMPORTS (crude only) ----
    if imports_df is not None and imports_cols:
        imp = imports_df.copy()
        per = _get(imp, imports_cols, "period")
        if per is not None: imp = imp[per == period]
        dest = _get(imp, imports_cols, "dest_area")
        val  = _get(imp, imports_cols, "value")
        if dest is None or val is None:
            raise ValueError("imports_cols must include 'dest_area' and 'value'")
        prod = _get(imp, imports_cols, "product")
        # keep crude only (if product present). If absent, treat all as crude.
        if prod is not None:
            imp = imp[prod.apply(lambda p: _is_crude(p, crude_keys))]
            # Preserve original product name for crude products
            imp = imp.assign(
                dest_area = dest.map(_norm_padd),
                value_bbl = val.apply(lambda v: _to_bbl(v, imports_unit)),
                origin_country = _get(imp, imports_cols, "origin_country"),
                product = prod,  # preserve original product names
            )
        else:
            # If no product columns_col, default to "crude"
            imp = imp.assign(
                dest_area = dest.map(_norm_padd),
                value_bbl = val.apply(lambda v: _to_bbl(v, imports_unit)),
                origin_country = _get(imp, imports_cols, "origin_country"),
                product = "crude",
            )
        imp = imp.groupby(["dest_area","origin_country","product"], dropna=False)["value_bbl"].sum().reset_index()
        imp = imp[imp["dest_area"].isin(PADD_CODES) & imp["value_bbl"].gt(0)]
        imp["direction"] = "import"; imp["origin_area"] = None; imp["dest_country"] = None; imp["period"] = period
        flows.append(imp[["direction","origin_area","dest_area","origin_country","dest_country","product","value_bbl","period"]])

    # ---- EXPORTS (crude + refined) ----
    if exports_df is not None and exports_cols:
        exp = exports_df.copy()
        per = _get(exp, exports_cols, "period")
        if per is not None: exp = exp[per == period]
        src = _get(exp, exports_cols, "origin_area")
        val = _get(exp, exports_cols, "value")
        if src is None or val is None:
            raise ValueError("exports_cols must include 'origin_area' and 'value'")
        prod = _get(exp, exports_cols, "product")
        # keep original product names, just filter for crude products if needed
        if prod is not None:
            exp = exp.assign(product = prod)  # preserve original product names
        else:
            exp = exp.assign(product = None)   # unknown → let it pass
        exp = exp.assign(
            origin_area = src.map(_norm_padd),
            value_bbl = val.apply(lambda v: _to_bbl(v, exports_unit)),
            dest_country = _get(exp, exports_cols, "dest_country"),
        )
        exp = exp.groupby(["origin_area","dest_country","product"], dropna=False)["value_bbl"].sum().reset_index()
        exp = exp[exp["origin_area"].isin(PADD_CODES) & exp["value_bbl"].gt(0)]
        exp["direction"] = "export"; exp["dest_area"] = None; exp["origin_country"] = None; exp["period"] = period
        flows.append(exp[["direction","origin_area","dest_area","origin_country","dest_country","product","value_bbl","period"]])

    # ---- INTER-PADD PAIRS (crude only) ----
    if inter_padd_pairs_df is not None and inter_pairs_cols:
        ip = inter_padd_pairs_df.copy()
        per = _get(ip, inter_pairs_cols, "period")
        if per is not None: ip = ip[per == period]
        src = _get(ip, inter_pairs_cols, "origin_area")
        dst = _get(ip, inter_pairs_cols, "dest_area")
        val = _get(ip, inter_pairs_cols, "value")
        if src is None or dst is None or val is None:
            raise ValueError("inter_pairs_cols must include 'origin_area','dest_area','value'")
        prod = _get(ip, inter_pairs_cols, "product")
        # keep crude only (if product present). If absent, assume crude.
        if prod is not None:
            ip = ip[prod.apply(lambda p: _is_crude(p, crude_keys))].copy()
            # Preserve original product name for crude products
            ip = ip.assign(
                origin_area = src.map(_norm_padd),
                dest_area   = dst.map(_norm_padd),
                value_bbl   = val.apply(lambda v: _to_bbl(v, inter_pairs_unit)),
                product     = prod,  # preserve original product names
            )
        else:
            # If no product columns_col, default to "crude"
            ip = ip.assign(
                origin_area = src.map(_norm_padd),
                dest_area   = dst.map(_norm_padd),
                value_bbl   = val.apply(lambda v: _to_bbl(v, inter_pairs_unit)),
                product     = "crude",
            )
        ip = ip.groupby(["origin_area","dest_area","product"], dropna=False)["value_bbl"].sum().reset_index()
        ip = ip[ip["origin_area"].isin(PADD_CODES) & ip["dest_area"].isin(PADD_CODES) & ip["value_bbl"].gt(0)]
        ip["direction"] = "inter_padd"; ip["origin_country"] = None; ip["dest_country"] = None; ip["period"] = period
        flows.append(ip[["direction","origin_area","dest_area","origin_country","dest_country","product","value_bbl","period"]])

    # ---- NET RECEIPTS (refined only → synthesize inter-PADD by product) ----
    if net_receipts_df is not None and net_receipts_cols:
        nr = net_receipts_df.copy()
        per = _get(nr, net_receipts_cols, "period")
        if per is not None: nr = nr[per == period]
        area = _get(nr, net_receipts_cols, "area")
        val  = _get(nr, net_receipts_cols, "value")
        prod = _get(nr, net_receipts_cols, "product")
        if area is None or val is None or prod is None:
            raise ValueError("net_receipts_cols must include 'area','value','product'")
        # Create a temporary columns_col for filtering but preserve original product names
        nr = nr.assign(
            area = area.map(_norm_padd),
            value_bbl = val.apply(lambda v: _to_bbl(v, net_receipts_unit)),
            product = prod,  # preserve original product names
            product_category = prod.apply(lambda p: "crude" if _is_crude(p, crude_keys) else "refined")
        )
        # keep refined only (avoid double-counting crude, since inter_padd_pairs already handled it)
        nrr = nr[(nr["area"].isin(PADD_CODES)) & (nr["product_category"] == "refined")].copy()
        # Drop the temporary category columns_col
        nrr = nrr.drop(columns=['product_category'])
        if not nrr.empty:
            pairs = _greedy_pairs_from_net(nrr, area_col="area", value_col="value_bbl", product_col="product")
            if not pairs.empty:
                pairs["direction"] = "inter_padd"
                pairs["origin_country"] = None
                pairs["dest_country"] = None
                pairs["period"] = period
                flows.append(pairs[["direction","origin_area","dest_area","origin_country","dest_country","product","value_bbl","period"]])

    # ---- STOCKS (crude + refined) ----
    if stocks_df is not None and stocks_cols:
        stk = stocks_df.copy()
        per = _get(stk, stocks_cols, "period")
        if per is not None: stk = stk[per == period]
        src = _get(stk, stocks_cols, "source_area")
        val = _get(stk, stocks_cols, "value")
        if src is None or val is None:
            raise ValueError("stocks_cols must include 'source_area' and 'value'")
        prod = _get(stk, stocks_cols, "product")
        # Get change columns if available
        prev_val = _get(stk, stocks_cols, "prev_period_value") 
        change_pct = _get(stk, stocks_cols, "change_pct")
        change_dir = _get(stk, stocks_cols, "change_direction")
        
        # keep original product names
        if prod is not None:
            stk = stk.assign(product = prod)  # preserve original product names
        else:
            stk = stk.assign(product = None)   # unknown → let it pass
        
        stk = stk.assign(
            source_area = src.map(_norm_padd),
            value_bbl = val.apply(lambda v: _to_bbl(v, stocks_unit)),
        )
        
        # Add change columns if available
        if prev_val is not None:
            stk = stk.assign(prev_period_value = prev_val)
        if change_pct is not None:
            stk = stk.assign(change_pct = change_pct)
        if change_dir is not None:
            stk = stk.assign(change_direction = change_dir)
        
        # Group by and preserve change columns by taking first value (since they should be the same for each group)
        group_cols = ["source_area", "product"]
        agg_dict = {"value_bbl": "sum"}
        if prev_val is not None:
            agg_dict["prev_period_value"] = "first"
        if change_pct is not None:
            agg_dict["change_pct"] = "first" 
        if change_dir is not None:
            agg_dict["change_direction"] = "first"
            
        stk = stk.groupby(group_cols, dropna=False).agg(agg_dict).reset_index()
        stk = stk[stk["source_area"].isin(PADD_CODES) & stk["value_bbl"].gt(0)]
        # For stocks, source_area represents where stocks are held
        stk["direction"] = "stocks"
        stk["origin_area"] = stk["source_area"]  # stocks are held at this location
        stk["dest_area"] = None                  # no destination for stocks
        stk["origin_country"] = None
        stk["dest_country"] = None
        stk["period"] = period
        
        # Include change columns if available
        base_cols = ["direction","origin_area","dest_area","origin_country","dest_country","product","value_bbl","period"]
        extra_cols = []
        if "prev_period_value" in stk.columns:
            extra_cols.append("prev_period_value")
        if "change_pct" in stk.columns:
            extra_cols.append("change_pct") 
        if "change_direction" in stk.columns:
            extra_cols.append("change_direction")
            
        flows.append(stk[base_cols + extra_cols])

    # ---- Stitch & sort ----
    if not flows:
        return pd.DataFrame(columns=["direction","origin_area","dest_area","origin_country","dest_country","product","value_bbl","period"])

    out = pd.concat(flows, ignore_index=True)
    cat = pd.Categorical(out["direction"], categories=["import","export","inter_padd","stocks"], ordered=True)
    out = out.assign(direction=cat).sort_values(["direction","product","origin_area","dest_area","value_bbl"], na_position="last").reset_index(drop=True)
    out["direction"] = out["direction"].astype(str)
    return out

# =============================================================================
# EIA DATA TRANSFORMATION FUNCTIONS
# =============================================================================

def transform_imports_by_district(eia_client: Optional[PETable] = None, data: Optional[pd.DataFrame] = None) -> pd.DataFrame:
    """
    Transform movements/imports/by_district data for build_flow_dataframe.
    
    Input structure: Index=period, columns with tuples like ('PADD1', 'MEX')
    Output structure: period, dest_area, origin_country, value
    
    Args:
        eia_client: Optional PETable instance with rename_key_cols=False
        data: Optional pre-loaded data to use instead of fetching from client
        
    Returns:
        DataFrame with columns: period, dest_area, origin_country, value
    """
    if data is None:
        if eia_client is None:
            raise ValueError("Either eia_client or data must be provided")
        data = eia_client.get_key('movements/imports/by_district')
    
    if data is None or data.empty:
        return pd.DataFrame(columns=['period', 'dest_area', 'origin_country', 'value'])
    
    # Reset index to get period as a columns_col
    df = data.reset_index()
    
    # Get only tuple columns (drop string columns like summary totals)
    tuple_cols = [col for col in df.columns if isinstance(col, tuple)]
    non_tuple_cols = [col for col in df.columns if not isinstance(col, tuple)]
    
    if not tuple_cols:
        return pd.DataFrame(columns=['period', 'dest_area', 'origin_country', 'value'])
    
    # Keep period and tuple columns
    df = df[non_tuple_cols + tuple_cols]
    
    # Melt the DataFrame to convert tuple columns to rows
    df = df.melt(id_vars=non_tuple_cols, value_vars=tuple_cols, 
                 var_name='padd_country', value_name='value')
    
    # Skip rows with NaN or zero values
    df = df.dropna(subset=['value'])
    df = df[df['value'] > 0]
    
    # Split the tuple into dest_area and origin_country
    df[['dest_area', 'origin_country']] = pd.DataFrame(df['padd_country'].tolist(), index=df.index)
    
    # Map country codes to full names
    country_mapping = {'CAN': 'Canada', 'MEX': 'Mexico', 'IRQ': 'Iraq', 'SAU': 'Saudi Arabia'}
    df['origin_country'] = df['origin_country'].map(country_mapping).fillna(df['origin_country'])
    
    # Normalize PADD names
    df['dest_area'] = df['dest_area'].apply(_norm_padd)
    
    # Add product field since this is crude imports only
    df['product'] = 'Crude Oil'
    
    # Clean up and return
    result = df[['period', 'dest_area', 'origin_country', 'product', 'value']].copy()
    result['period'] = pd.to_datetime(result['period']).dt.strftime('%Y-%m')
    
    return result


def transform_exports_by_district(eia_client: Optional[PETable] = None, data: Optional[pd.DataFrame] = None) -> pd.DataFrame:
    """
    Transform movements/exports/by_district data for build_flow_dataframe.
    
    Input structure: Index=period, MultiIndex columns (area-name, product) 
    Output structure: period, origin_area, product, value
    
    Args:
        eia_client: Optional PETable instance with rename_key_cols=False
        data: Optional pre-loaded data to use instead of fetching from client
        
    Returns:
        DataFrame with columns: period, origin_area, product, value
    """
    if data is None:
        if eia_client is None:
            raise ValueError("Either eia_client or data must be provided")
        data = eia_client.get_key('movements/exports/by_district')
    
    if data is None or data.empty:
        return pd.DataFrame(columns=['period', 'origin_area', 'product', 'value'])
    
    # Reset index to get period as a columns_col
    df = data.reset_index()
    
    # Check if we have MultiIndex columns
    if hasattr(data.columns, 'nlevels') and data.columns.nlevels > 1:
        # Melt MultiIndex columns - need to stack first
        df = df.set_index('period')
        df = df.stack(level=[0, 1], dropna=True).reset_index()
        df.columns = ['period', 'origin_area', 'product', 'value']
    else:
        return pd.DataFrame(columns=['period', 'origin_area', 'product', 'value'])
    
    # Skip rows with NaN or zero values
    df = df.dropna(subset=['value'])
    df = df[df['value'] > 0]
    
    # Normalize PADD names
    df['origin_area'] = df['origin_area'].apply(_norm_padd)
    
    # Map product codes to names
    product_mapping = {
        'EPC0': 'Crude Oil',
        'EPD0': 'Distillate',
        'EPJK': 'Kerosene',
        'EPM0F': 'Motor Gasoline'
    }
    df['product'] = df['product'].map(product_mapping).fillna(df['product'])
    
    # Clean up and return
    result = df[['period', 'origin_area', 'product', 'value']].copy()
    result['period'] = pd.to_datetime(result['period']).dt.strftime('%Y-%m')
    
    return result


def transform_net_receipts(eia_client: Optional[PETable] = None, data: Optional[pd.DataFrame] = None) -> pd.DataFrame:
    """
    Transform movements/net data for build_flow_dataframe.
    
    Input structure: No index, MultiIndex columns with ('area', 'product') and ('period', '') columns_col
    Output structure: period, area, product, value
    
    Args:
        eia_client: Optional PETable instance with rename_key_cols=False
        data: Optional pre-loaded data to use instead of fetching from client
        
    Returns:
        DataFrame with columns: period, area, product, value
    """
    if data is None:
        if eia_client is None:
            raise ValueError("Either eia_client or data must be provided")
        data = eia_client.get_key('movements/net')
    
    if data is None or data.empty:
        return pd.DataFrame(columns=['period', 'area', 'product', 'value'])
    
    # Extract period columns_col first
    period_col = data[('period', '')]
    
    # Get non-period columns (area-product combinations)
    area_product_cols = [col for col in data.columns if col != ('period', '')]
    
    # Stack the MultiIndex columns to convert to long format
    df = data[area_product_cols].stack(level=[0, 1], dropna=True).reset_index()
    df.columns = ['row_index', 'area', 'product', 'value']
    
    # Add period back using the row_index to match
    df['period'] = period_col.iloc[df['row_index']].values
    
    # Skip rows with NaN or zero values
    df = df.dropna(subset=['value'])
    
    # Normalize PADD names
    df['area'] = df['area'].apply(_norm_padd)
    
    # Filter out rows where area normalization failed
    df = df[df['area'].notna()]
    
    # Filter out empty product names (from period columns_col)
    df = df[df['product'] != '']
    
    # Map product codes to names
    product_mapping = {
        'EPD0': 'Distillate',
        'EPJK': 'Kerosene', 
        'EPM0F': 'Motor Gasoline'
    }
    df['product'] = df['product'].map(product_mapping).fillna(df['product'])
    
    # Clean up and return
    result = df[['period', 'area', 'product', 'value']].copy()
    result['period'] = pd.to_datetime(result['period']).dt.strftime('%Y-%m')
    
    return result


def transform_crude_movements(eia_client: Optional[PETable] = None, data: Optional[pd.DataFrame] = None) -> pd.DataFrame:
    """
    Transform movements/crude_movements data for build_flow_dataframe.
    
    Input structure: Single level columns with tuples like ('PADD1', 'PADD 3')
    Output structure: period, origin_area, dest_area, value
    
    Args:
        eia_client: Optional PETable instance with rename_key_cols=False
        data: Optional pre-loaded data to use instead of fetching from client
        
    Returns:
        DataFrame with columns: period, origin_area, dest_area, value
    """
    if data is None:
        if eia_client is None:
            raise ValueError("Either eia_client or data must be provided")
        data = eia_client.get_key('movements/crude_movements')
    
    if data is None or data.empty:
        return pd.DataFrame(columns=['period', 'origin_area', 'dest_area', 'value'])
    
    # Reset index to get period as a columns_col
    df = data.reset_index()
    
    # Drop non-tuple columns (summary columns, units, etc.)
    tuple_cols = [col for col in df.columns if isinstance(col, tuple)]
    df = df[['period'] + tuple_cols]
    
    # Melt the DataFrame to convert tuple columns to rows
    df = df.melt(id_vars=['period'], var_name='origin_dest', value_name='value')
    
    # Skip rows with NaN or zero values
    df = df.dropna(subset=['value'])
    df = df[df['value'] > 0]
    
    # Split the tuple into origin_area and dest_area
    df[['origin_area', 'dest_area']] = pd.DataFrame(df['origin_dest'].tolist(), index=df.index)
    
    # Normalize PADD names for both origin and destination
    df['origin_area'] = df['origin_area'].apply(_norm_padd)
    df['dest_area'] = df['dest_area'].apply(_norm_padd)
    
    # Filter out rows where normalization failed
    df = df[df['origin_area'].notna() & df['dest_area'].notna()]
    
    # Add product field since this is crude movements only
    df['product'] = 'Crude Oil'
    
    # Clean up and return
    result = df[['period', 'origin_area', 'dest_area', 'product', 'value']].copy()
    result['period'] = pd.to_datetime(result['period']).dt.strftime('%Y-%m')
    
    return result


def transform_product_stocks(eia_client: Optional[PETable] = None, data: Optional[pd.DataFrame] = None) -> pd.DataFrame:
    """
    Transform supply/product_stocks data for build_flow_dataframe.
    
    Input structure: Index=period, MultiIndex columns (source_area, product)
    Output structure: period, source_area, product, value
    
    Args:
        eia_client: Optional PETable instance with rename_key_cols=False
        data: Optional pre-loaded data to use instead of fetching from client
        
    Returns:
        DataFrame with columns: period, source_area, product, value
    """
    if data is None:
        if eia_client is None:
            raise ValueError("Either eia_client or data must be provided")
        data = eia_client.get_key('supply/product_stocks')
    
    if data is None or data.empty:
        return pd.DataFrame(columns=['period', 'source_area', 'product', 'value'])
    
    # Reset index to get period as a columns_col
    df = data.reset_index()
    
    # Check if we have MultiIndex columns
    if hasattr(data.columns, 'nlevels') and data.columns.nlevels > 1:
        # Melt MultiIndex columns - need to stack first
        df = df.set_index('period')
        df = df.stack(level=[0, 1], dropna=True).reset_index()
        df.columns = ['period', 'source_area', 'product', 'value']
    else:
        return pd.DataFrame(columns=['period', 'source_area', 'product', 'value'])
    
    # Convert value to numeric and skip rows with NaN or zero values
    df['value'] = pd.to_numeric(df['value'], errors='coerce')
    df = df.dropna(subset=['value'])
    df = df[df['value'] > 0]
    
    # Normalize PADD names
    df['source_area'] = df['source_area'].apply(_norm_padd)
    
    # Filter out rows where area normalization failed
    df = df[df['source_area'].notna()]
    
    # Map product codes to names
    product_mapping = {
        'EPC0': 'Crude Oil',
        'EPD0': 'Distillate',
        'EPJK': 'Kerosene',
        'EPM0F': 'Motor Gasoline'
    }
    df['product'] = df['product'].map(product_mapping).fillna(df['product'])
    
    # Clean up and return
    result = df[['period', 'source_area', 'product', 'value']].copy()
    result['period'] = pd.to_datetime(result['period']).dt.strftime('%Y-%m')
    
    return result


def get_flow_dataframes_from_eia(period: str, eia_client: Optional[PETable] = None) -> dict:
    """
    Get all transformed flow dataframes from EIA data for a specific period.
    
    Args:
        period: Period in 'YYYY-MM' format
        eia_client: Optional PETable instance. If None, creates one with rename_key_cols=False
        
    Returns:
        Dictionary with keys: imports_df, exports_df, net_receipts_df, crude_movements_df, stocks_df
    """
    if eia_client is None:
        eia_client = PETable(rename_key_cols=False)
    
    # Transform all datasets
    imports_df = transform_imports_by_district(eia_client)
    exports_df = transform_exports_by_district(eia_client)
    net_receipts_df = transform_net_receipts(eia_client)
    crude_movements_df = transform_crude_movements(eia_client)
    stocks_df = transform_product_stocks(eia_client)
    
    # Filter to specific period
    period_filter = lambda df: df[df['period'] == period] if not df.empty else df
    
    return {
        'imports_df': period_filter(imports_df),
        'exports_df': period_filter(exports_df), 
        'net_receipts_df': period_filter(net_receipts_df),
        'crude_movements_df': period_filter(crude_movements_df),
        'stocks_df': period_filter(stocks_df)
    }


# =============================================================================
# UNIFIED FLOW BUILDER CLASS
# =============================================================================

class EIAFlowBuilder:

    """
    A reusable class that combines EIA data transformation and flow building functionality.
    
    This class handles:
    1. Loading data from EIATable with different columns_col structures
    2. Transforming MultiIndex/tuple columns to standard formats
    3. Building unified flow DataFrames compatible with visualization
    4. Providing columns_col mappings for different data types
    
    Usage:
        builder = EIAFlowBuilder()
        flow_df = builder.build_period_flows("2024-01")
        
        # Or get individual transformed datasets
        imports_df = builder.get_imports_data()
        exports_df = builder.get_exports_data()
    """
    
    def __init__(self, eia_client: Optional[PETable] = None):
        """
        Initialize the flow builder and preload all flow data.
        
        Args:
            eia_client: Optional PETable instance. If None, creates one with rename_key_cols=False
        """
        self.eia_client = eia_client if eia_client is not None else PETable(rename_key_cols=False)
        
        # Country mapping for imports
        self.country_mapping = {
            'CAN': 'Canada', 
            'MEX': 'Mexico', 
            'IRQ': 'Iraq', 
            'SAU': 'Saudi Arabia'
        }
        
        # Product mapping for exports/refined products
        self.product_mapping = {
            'EPC0': 'Crude Oil',
            'EPD0': 'Distillate',
            'EPJK': 'Kerosene',
            'EPM0F': 'Motor Gasoline'
        }
        
        # Column mappings for build_flow_dataframe
        self.column_mappings = {
            'imports': {
                "period": "period",
                "dest_area": "dest_area", 
                "value": "value",
                "origin_country": "origin_country",
                "product": "product"
            },
            'exports': {
                "period": "period",
                "origin_area": "origin_area",
                "value": "value",
                "product": "product"
            },
            'net_receipts': {
                "period": "period",
                "area": "area",
                "value": "value",
                "product": "product"
            },
            'crude_movements': {
                "period": "period",
                "origin_area": "origin_area",
                "dest_area": "dest_area", 
                "value": "value",
                "product": "product"
            },
            'stocks': {
                "period": "period",
                "source_area": "source_area",
                "value": "value",
                "product": "product",
                "prev_period_value": "prev_period_value",
                "change_pct": "change_pct",
                "change_direction": "change_direction"
            }
        }
        
        # Preload all flow data into self.data
        self.data = self._load_all_flow_data()
    
    def _load_all_flow_data(self) -> dict:
        """
        Load and transform all flow data from EIA client.
        
        Returns:
            Dictionary with keys: imports_df, exports_df, net_receipts_df, crude_movements_df, stocks_df
        """
        # Load raw data from EIA client
        raw_data = {
            'imports': self.eia_client.get_key('movements/imports/by_district'),
            'exports': self.eia_client.get_key('movements/exports/by_district'),
            'net': self.eia_client.get_key('movements/net'),
            'crude_movements': self.eia_client.get_key('movements/crude_movements'),
            'product_stocks': self.eia_client.get_key('supply/product_stocks')
        }
        
        # Transform raw data using transform functions
        return {
            'imports_df': transform_imports_by_district(data=raw_data['imports']),
            'exports_df': transform_exports_by_district(data=raw_data['exports']),
            'net_receipts_df': transform_net_receipts(data=raw_data['net']),
            'crude_movements_df': transform_crude_movements(data=raw_data['crude_movements']),
            'stocks_df': transform_product_stocks(data=raw_data['product_stocks'])
        }
    
    def get_imports_data(self) -> pd.DataFrame:
        """Get transformed imports by district data from cached data."""
        return self.data['imports_df'].copy()
    
    def get_exports_data(self) -> pd.DataFrame:
        """Get transformed exports by district data from cached data.""" 
        return self.data['exports_df'].copy()
    
    def get_net_receipts_data(self) -> pd.DataFrame:
        """Get transformed net receipts data from cached data."""
        return self.data['net_receipts_df'].copy()
    
    def get_crude_movements_data(self) -> pd.DataFrame:
        """Get transformed crude movements data from cached data."""
        return self.data['crude_movements_df'].copy()
    
    def get_stocks_data(self) -> pd.DataFrame:
        """Get transformed product stocks data from cached data."""
        return self.data['stocks_df'].copy()
    
    def _calculate_stock_changes(self, stocks_df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate period-over-period changes for stock data and add to dataframe.
        
        Args:
            stocks_df: Stock dataframe with columns: period, source_area, product, value
            
        Returns:
            Enhanced dataframe with additional columns: prev_period_value, change_pct, change_direction
        """
        if stocks_df.empty or 'period' not in stocks_df.columns:
            return stocks_df
        
        # Sort by source_area, product, and period
        df_sorted = stocks_df.sort_values(['source_area', 'product', 'period']).copy()
        
        # Calculate previous period values within each group
        df_sorted['prev_period_value'] = df_sorted.groupby(['source_area', 'product'])['value'].shift(1)
        
        # Calculate percentage change
        df_sorted['change_pct'] = (
            (df_sorted['value'] - df_sorted['prev_period_value']) / df_sorted['prev_period_value'] * 100
        ).fillna(0)
        
        # Add change direction symbols
        df_sorted['change_direction'] = df_sorted['change_pct'].apply(
            lambda x: "↑" if x > 0 else "↓" if x < 0 else "→"
        )
        
        # Return in original order
        return df_sorted.sort_index()
    
    
    def build_period_flows(
        self, 
        period: str, 
        include_imports: bool = True,
        include_exports: bool = True, 
        include_inter_padd: bool = True,
        include_stocks: bool = True,
        use_crude_movements: bool = True,
        product_filter: Optional[Iterable[str]] = None
    ) -> pd.DataFrame:
        """
        Build a unified flow DataFrame for a specific period.
        
        Args:
            period: Period in 'YYYY-MM' format
            include_imports: Include import flows
            include_exports: Include export flows
            include_inter_padd: Include inter-PADD flows
            include_stocks: Include stock levels
            use_crude_movements: If True, use crude movements data; if False, use net receipts fallback
            product_filter: Optional list of products to include
            
        Returns:
            Unified flow DataFrame with direction, origin_area, dest_area, etc.
        """
        # Filter cached data by period
        flow_data = {}
        for key, df in self.data.items():
            if not df.empty and 'period' in df.columns:
                flow_data[key] = df[df['period'] == period]
            else:
                flow_data[key] = df.copy() if not df.empty else df
        
        # Prepare arguments for build_flow_dataframe
        kwargs = {'period': period}
        
        if include_imports and not flow_data['imports_df'].empty:
            kwargs['imports_df'] = flow_data['imports_df']
            kwargs['imports_cols'] = self.column_mappings['imports']
            
        if include_exports and not flow_data['exports_df'].empty:
            kwargs['exports_df'] = flow_data['exports_df'] 
            kwargs['exports_cols'] = self.column_mappings['exports']
        
        if include_inter_padd:
            if use_crude_movements and not flow_data['crude_movements_df'].empty:
                # Use crude movements data (pairwise flows for crude)
                kwargs['inter_padd_pairs_df'] = flow_data['crude_movements_df']
                kwargs['inter_pairs_cols'] = self.column_mappings['crude_movements']
            
            # Always include net receipts for refined products (if available)
            if not flow_data['net_receipts_df'].empty:
                # Include net receipts for refined products (will be balanced into pairs)
                kwargs['net_receipts_df'] = flow_data['net_receipts_df']
                kwargs['net_receipts_cols'] = self.column_mappings['net_receipts']
        
        if include_stocks and not flow_data['stocks_df'].empty:
            # Calculate stock changes using full dataset, then filter by period
            stocks_with_changes = self._calculate_stock_changes(self.data['stocks_df'])
            if not stocks_with_changes.empty and 'period' in stocks_with_changes.columns:
                kwargs['stocks_df'] = stocks_with_changes[stocks_with_changes['period'] == period]
            else:
                kwargs['stocks_df'] = flow_data['stocks_df']
            kwargs['stocks_cols'] = self.column_mappings['stocks']
        
        if product_filter:
            kwargs['product_filter'] = product_filter
            
        return build_flow_dataframe(**kwargs)

    def generate_period_map(self, period):

        padd_map = pm.make_padd_choropleth('US Petroleum product flows')
        flow_df = self.build_period_flows(period)
        flow_map = pm.add_flow_traces(padd_map,flow_df, period)

        return flow_map
    
    def get_available_periods(self) -> list:
        """
        Get list of available periods from the data.
        
        Returns:
            Sorted list of available periods in 'YYYY-MM' format
        """
        all_periods = set()
        
        for dataset in [self.get_imports_data(), self.get_exports_data(), 
                       self.get_net_receipts_data(), self.get_crude_movements_data(),
                       self.get_stocks_data()]:
            if not dataset.empty and 'period' in dataset.columns:
                all_periods.update(dataset['period'].unique())
        
        return sorted(list(all_periods))
    
    def get_summary_stats(self, period: Optional[str] = None) -> dict:
        """
        Get summary statistics for the datasets.
        
        Args:
            period: Optional period to filter stats
            
        Returns:
            Dictionary with summary statistics for each dataset
        """
        flow_data = self.get_all_flow_data(period)
        
        stats = {}
        for name, df in flow_data.items():
            if not df.empty:
                stats[name] = {
                    'records': len(df),
                    'date_range': f"{df['period'].min()} to {df['period'].max()}" if 'period' in df.columns else 'N/A',
                    'total_value': df['value'].sum() if 'value' in df.columns else 0,
                    'avg_value': df['value'].mean() if 'value' in df.columns else 0
                }
            else:
                stats[name] = {'records': 0, 'date_range': 'No data', 'total_value': 0, 'avg_value': 0}
        
        return stats
    
    def build_comparison_flows(self, periods: list, **kwargs) -> pd.DataFrame:
        """
        Build flow DataFrames for multiple periods for comparison.
        
        Args:
            periods: List of periods in 'YYYY-MM' format
            **kwargs: Additional arguments passed to build_period_flows
            
        Returns:
            Combined DataFrame with flows from all periods
        """
        all_flows = []
        
        for period in periods:
            try:
                period_flows = self.build_period_flows(period, **kwargs)
                if not period_flows.empty:
                    all_flows.append(period_flows)
            except Exception as e:
                print(f"Warning: Failed to build flows for period {period}: {e}")
        
        if all_flows:
            return pd.concat(all_flows, ignore_index=True)
        else:
            return pd.DataFrame(columns=["direction","origin_area","dest_area","origin_country","dest_country","product","value_bbl","period"])
    
    def export_flow_data(self, period: str, filepath: str, format: str = 'csv') -> bool:
        """
        Export flow data to file.
        
        Args:
            period: Period in 'YYYY-MM' format
            filepath: Output file path
            format: Export format ('csv', 'excel', 'json')
            
        Returns:
            True if successful, False otherwise
        """
        try:
            flow_df = self.build_period_flows(period)
            
            if flow_df.empty:
                print(f"No flow data available for period {period}")
                return False
            
            if format.lower() == 'csv':
                flow_df.to_csv(filepath, index=False)
            elif format.lower() == 'excel':
                flow_df.to_excel(filepath, index=False)
            elif format.lower() == 'json':
                flow_df.to_json(filepath, orient='records', date_format='iso')
            else:
                raise ValueError(f"Unsupported format: {format}")
                
            print(f"Successfully exported {len(flow_df)} flow records to {filepath}")
            return True
            
        except Exception as e:
            print(f"Error exporting flow data: {e}")
            return False
