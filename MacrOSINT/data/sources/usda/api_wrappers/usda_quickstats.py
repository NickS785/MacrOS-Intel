
"""
usda_quickstats.py
-------------------
A lightweight Python wrapper for the USDA/NASS Quick Stats API:
https://quickstats.nass.usda.gov/api

Features
--------
- Endpoints covered: /api_GET, /get_counts, /get_param_values
- Simple, typed client with requests.Session reuse and retries
- Helper methods to return pandas.DataFrame (if pandas is installed)
- Automatic chunking by year to stay under the 50,000 row limit
- Environment-variable support for API key: NASS_API_KEY or USDA_NASS_API_KEY
- Friendly error messages with API response context

Quick Start
-----------
from usda_quickstats import QuickStatsClient
qs = QuickStatsClient(api_key="YOUR_KEY")
data = qs.query(commodity_desc="CORN", state_alpha="IA", year__GE=2019, agg_level_desc="STATE")
len(data), data[0].keys()

Or as a DataFrame (requires pandas):
df = qs.query_df(commodity_desc="CORN", state_alpha="IA", year__GE=2019, agg_level_desc="STATE")
print(df.head())

For large queries, use fetch_all() to automatically split by year:
big = qs.fetch_all(commodity_desc="CORN", agg_level_desc="COUNTY")
print(len(big))
"""
from __future__ import annotations

import os
import time
import logging
import asyncio
import aiohttp
from typing import Dict, Any, List, Optional, Iterable, Union
import requests
from MacrOSINT import config
from dotenv import load_dotenv

try:
    import pandas as pd  # optional
except Exception:  # pragma: no cover
    pd = None

__all__ = ["QuickStatsClient", "QuickStatsError"]

log = logging.getLogger(__name__)
if not config.NASS_TOKEN:
    if not load_dotenv(config.DOT_ENV):
        print("Failed to load API key from environment")
    else:
        NASS_TOKEN = os.getenv('NASS_TOKEN', '')
else:
    NASS_TOKEN = config.NASS_TOKEN


class QuickStatsError(RuntimeError):
    """Raised when the Quick Stats API returns an error or an invalid response."""

class QuickStatsClient:
    """Client for the USDA/NASS Quick Stats API.

    Docs: https://quickstats.nass.usda.gov/api

    Parameters
    ----------
    api_key : str | None
        Your Quick Stats API key. If omitted, the client will try environment
        variables ``NASS_API_KEY`` then ``USDA_NASS_API_KEY``.
    base_url : str
        Base URL for the API. Defaults to ``https://quickstats.nass.usda.gov/api``.
    timeout : float
        Per-request timeout in seconds. Default 30.
    max_retries : int
        Number of simple retry attempts for transient HTTP errors (>=500) and
        certain network errors. Default 3.
    backoff : float
        Seconds to sleep between retries. Default 1.25.
    session : requests.Session | None
        Optionally provide your own Session. Otherwise one is created.
    """

    # The API enforces a maximum of 50,000 rows per /api_GET call.
    HARD_ROW_LIMIT = 50_000

    # Valid parameters (from the docs). Useful for validation and auto-filtering.
    # Validation is *soft*; the API accepts many combinations and operators.
    VALID_PARAMS = {
        "source_desc","sector_desc","group_desc","commodity_desc","class_desc",
        "prodn_practice_desc","util_practice_desc","statisticcat_desc","unit_desc",
        "short_desc","domain_desc","domaincat_desc","agg_level_desc","state_ansi",
        "state_fips_code","state_alpha","state_name","asd_code","asd_desc","county_ansi",
        "county_code","county_name","region_desc","zip_5","watershed_code","watershed_desc",
        "congr_district_code","country_code","country_name","location_desc","year",
        "freq_desc","begin_code","end_code","reference_period_desc","week_ending","load_time",
        "Value"
    }

    # Supported operators the API allows as suffixes (e.g., year__GE=2010).
    VALID_OPERATORS = {"__LE","__LT","__GT","__GE","__LIKE","__NOT_LIKE","__NE"}

    def __init__(
        self,
        api_key: Optional[str] = None,
        *,
        base_url: str = "https://quickstats.nass.usda.gov/api",
        timeout: float = 30.0,
        max_retries: int = 3,
        backoff: float = 1.25,
        session: Optional[requests.Session] = None,
    ) -> None:
        self.api_key = NASS_TOKEN or api_key
        if not self.api_key:
            raise QuickStatsError(
                "An API key is required. Get one at https://quickstats.nass.usda.gov/api "
                "and pass api_key=... or set NASS_TOKEN."
            )
        self.base_url = base_url.rstrip("/")
        self.timeout = float(timeout)
        self.max_retries = int(max_retries)
        self.backoff = float(backoff)
        self.session = session or requests.Session()
        self.session.headers.update({
            "User-Agent": "usda-quickstats-python/0.1 (+https://quickstats.nass.usda.gov/api)"
        })

    # ------------------------- High-level helpers -------------------------
    def query(self, **filters: Any) -> List[Dict[str, Any]]:
        """Fetch data records via /api_GET as a list of dicts.

        Examples
        --------
        >>> qs.query(commodity_desc="CORN", state_alpha="IA", year__GE=2019)
        """
        return self.api_get(filters, fmt="JSON")

    def query_df(self, **filters: Any):
        """Fetch data via /api_GET and return as a pandas DataFrame (if available)."""
        data = self.query(**filters)
        if pd is None:
            raise QuickStatsError("pandas is not installed. Try `pip install pandas`.")
        return pd.DataFrame(data)

    def get_counts(self, **filters: Any) -> int:
        """Return the number of rows that match filters using /get_counts."""
        payload = self._with_key(filters)
        url = f"{self.base_url}/get_counts/"
        resp = self._request("GET", url, params=payload)
        try:
            obj = resp.json()
            return int(obj.get("count", 0))
        except Exception as e:
            raise QuickStatsError(f"Invalid /get_counts JSON: {resp.text[:400]}") from e

    def get_param_values(self, param: str, **filters: Any) -> List[str]:
        """Return all possible values for a parameter via /get_param_values.

        You can optionally pass additional filters to scope the values.
        Example:
        >>> qs.get_param_values("year", commodity_desc="CORN", state_alpha="IA")
        """
        if not param:
            raise QuickStatsError("param is required for get_param_values")
        payload = self._with_key({"param": param, **filters})
        url = f"{self.base_url}/get_param_values/"
        resp = self._request("GET", url, params=payload)
        try:
            obj = resp.json()
            # Shape is {"param_name": ["A","B",...]} or {"error":"..."} on failure
            if "error" in obj:
                raise QuickStatsError(f"/get_param_values error: {obj['error']}")
            # Return the first list value found
            for k, v in obj.items():
                if isinstance(v, list):
                    return [str(x) for x in v]
            raise QuickStatsError(f"Unexpected /get_param_values JSON: {obj}")
        except ValueError as e:
            raise QuickStatsError(f"Invalid /get_param_values JSON: {resp.text[:400]}") from e

    def fetch_all(
        self,
        *,
        year_batch_size: int = 5,
        sleep: float = 0.0,
        **filters: Any,
    ) -> List[Dict[str, Any]]:
        """Fetch *all* rows for a query by chunking on years to avoid the 50k limit.

        Strategy
        --------
        1) Determine candidate years using get_param_values('year', **filters_except_year).
        2) Group contiguous years into batches of at most `year_batch_size` years.
        3) For each batch, check /get_counts; if > 50k, split further until <= 50k.
        4) Call /api_GET for each batch and extend the results.

        Notes
        -----
        - If you already pass an explicit year filter (e.g., year=2024 or year__GE/__LE)
          we still use it to derive the candidate list.
        - If you pass `freq_desc="WEEKLY"` and use weeks, the API still requires year-based
          filters; chunking by year remains appropriate.
        """
        # Build a candidate list of years based on filters
        years = self._candidate_years(**filters)
        if not years:
            # Fallback: try without scoping
            years = sorted(set(self.get_param_values("year")))
        results: List[Dict[str, Any]] = []
        # Remove any explicit year filters when chunking; we will add them per batch.
        scope_filters = {k: v for k, v in filters.items() if not k.startswith("year")}

        def batches(seq: List[int], k: int) -> Iterable[List[int]]:
            for i in range(0, len(seq), k):
                yield seq[i:i+k]

        for chunk in batches(years, year_batch_size):
            # Adaptively split until count <= limit
            lo, hi = chunk[0], chunk[-1]
            subfilters = {**scope_filters}
            if lo == hi:
                subfilters["year"] = lo
            else:
                subfilters["year__GE"] = lo
                subfilters["year__LE"] = hi
            count = self.get_counts(**subfilters)
            if count == 0:
                continue
            # If too big, split the chunk
            if count > self.HARD_ROW_LIMIT and len(chunk) > 1:
                # Split recursively by halving the year range until under the limit
                for sub in self._split_until_under_limit(chunk, scope_filters):
                    results.extend(self.api_get(sub, fmt="JSON"))
                    if sleep:
                        time.sleep(sleep)
            else:
                results.extend(self.api_get(subfilters, fmt="JSON"))
                if sleep:
                    time.sleep(sleep)
        return results

    # ------------------------- Core HTTP methods -------------------------
    def api_get(self, params: Dict[str, Any], fmt: str = "JSON") -> Union[List[Dict[str, Any]], str]:
        """Call /api_GET with filters.

        Parameters
        ----------
        params : dict
            Query parameters including filters like commodity_desc, state_alpha, year, etc.
            You may use operators like ``year__GE=2010`` or ``short_desc__LIKE="CORN%"``.
        fmt : {"JSON","CSV","XML","json","csv","xml"}
            Desired format. JSON returns a python list of dicts from the `data` field.
            CSV or XML return raw text.
        """
        fmt_upper = fmt.upper()
        if fmt_upper not in {"JSON", "CSV", "XML"}:
            raise QuickStatsError("fmt must be one of JSON, CSV, XML")
        payload = self._with_key(params)
        payload["format"] = fmt_upper
        url = f"{self.base_url}/api_GET/"
        resp = self._request("GET", url, params=payload)
        if fmt_upper == "JSON":
            try:
                obj = resp.json()
                if "error" in obj:
                    raise QuickStatsError(f"/api_GET error: {obj['error']}")
                data = obj.get("data")
                if not isinstance(data, list):
                    raise QuickStatsError(f"Unexpected /api_GET JSON: {obj}")
                # Normalize keys: some responses use 'Value', some 'Value ' - be tolerant
                normed = []
                for rec in data:
                    if not isinstance(rec, dict):
                        continue
                    # Strip whitespace from keys
                    normed.append({str(k).strip(): v for k, v in rec.items()})
                return normed
            except ValueError as e:
                raise QuickStatsError(f"Invalid /api_GET JSON: {resp.text[:400]}") from e
        else:
            return resp.text

    def _request(self, method: str, url: str, *, params: Optional[Dict[str, Any]] = None) -> requests.Response:
        last_exc: Optional[Exception] = None
        for attempt in range(1, self.max_retries + 2):
            try:
                resp = self.session.request(method, url, params=params, timeout=self.timeout)
                # Basic retry on 5xx
                if resp.status_code >= 500 and attempt <= self.max_retries:
                    time.sleep(self.backoff * attempt)
                    continue
                # Raise for HTTP errors
                if resp.status_code >= 400:
                    # Try to show API-provided error body
                    message = resp.text.strip()
                    raise QuickStatsError(f"HTTP {resp.status_code} for {url}: {message[:400]}")
                return resp
            except (requests.ConnectionError, requests.Timeout) as e:
                last_exc = e
                if attempt <= self.max_retries:
                    time.sleep(self.backoff * attempt)
                    continue
                raise QuickStatsError(f"Network error contacting Quick Stats API: {e}") from e
        # Should not reach here
        assert False, f"Unreachable; last_exc={last_exc}"

    # ------------------------- Utilities -------------------------
    def _with_key(self, params: Dict[str, Any]) -> Dict[str, Any]:
        payload = {"key": self.api_key}
        for k, v in params.items():
            if v is None:
                continue
            # Accept sequences for a param (API treats repeated params as an implicit OR)
            if isinstance(v, (list, tuple, set)):
                # Convert set to list for stability
                for item in list(v):
                    payload.setdefault(k, [])
                    payload[k].append(self._to_str(item))
            else:
                payload[k] = self._to_str(v)
        return payload

    @staticmethod
    def _to_str(v: Any) -> str:
        # Preserve booleans as lowercase strings if needed
        if isinstance(v, bool):
            return str(v).lower()
        return str(v)

    def _split_until_under_limit(self, year_chunk: List[int], scope_filters: Dict[str, Any]) -> Iterable[Dict[str, Any]]:
        """Yield subqueries (dicts) that each return <= HARD_ROW_LIMIT rows."""
        def recurse(yrs: List[int]):
            if not yrs:
                return
            lo, hi = yrs[0], yrs[-1]
            sub = {**scope_filters}
            if lo == hi:
                sub["year"] = lo
            else:
                sub["year__GE"] = lo
                sub["year__LE"] = hi
            cnt = self.get_counts(**sub)
            if cnt <= self.HARD_ROW_LIMIT:
                yield sub
            else:
                if len(yrs) == 1:
                    # Single year too large - no safe way to chunk further without additional filter.
                    # We still yield and let the API error; caller can add more filters (e.g., state or county).
                    log.warning("Single-year query still exceeds limit (%s). Consider adding more filters.", cnt)
                    yield sub
                else:
                    mid = len(yrs) // 2
                    for q in recurse(yrs[:mid]):
                        yield q
                    for q in recurse(yrs[mid:]):
                        yield q
        yield from recurse(year_chunk)

    def _candidate_years(self, **filters: Any) -> List[int]:
        """Build a sorted list of candidate years from filters + API values."""
        # If caller already gave an explicit list of years
        if "year" in filters and isinstance(filters["year"], (list, tuple, set)):
            return sorted({int(y) for y in filters["year"]})
        # Build scope without year filters for get_param_values("year")
        scope = {k: v for k, v in filters.items() if not k.startswith("year")}
        vals = self.get_param_values("year", **scope)
        # Apply local filtering for operators if provided
        years = sorted({int(x) for x in vals if str(x).isdigit()})
        # Apply operator bounds locally (year__GE/LE)
        ge = filters.get("year__GE")
        le = filters.get("year__LE")
        if ge is not None:
            years = [y for y in years if y >= int(ge)]
        if le is not None:
            years = [y for y in years if y <= int(le)]
        # If year is set to a scalar, intersect
        if "year" in filters and isinstance(filters["year"], (str, int)):
            y = int(filters["year"])
            years = [yy for yy in years if yy == y]
        return years

    # ------------------------- Convenience -------------------------
    @staticmethod
    def coerce_value_column(records: List[Dict[str, Any]], *, column: str = "Value") -> None:
        """In-place convert the 'Value' columns_col values to numeric where possible.

        Non-numeric codes like '(D)', '(Z)' become None.
        """
        for rec in records:
            if column in rec and rec[column] is not None:
                s = str(rec[column]).replace(",", "").strip()
                try:
                    rec[column] = float(s)
                except ValueError:
                    rec[column] = None

    def query_df_numeric(self, **filters: Any):
        """Like query_df but attempts to coerce the 'Value' columns_col to numeric."""
        data = self.query(**filters)
        self.coerce_value_column(data, column="Value")
        if pd is None:
            raise QuickStatsError("pandas is not installed. Try `pip install pandas`.")
        return pd.DataFrame(data)

    # ------------------------- Async Methods -------------------------
    async def query_async(self, **filters: Any) -> List[Dict[str, Any]]:
        """Async version of query() for concurrent requests."""
        return await self.api_get_async(filters, fmt="JSON")

    async def query_df_numeric_async(self, **filters: Any):
        """Async version of query_df_numeric() for concurrent requests."""
        data = await self.query_async(**filters)
        self.coerce_value_column(data, column="Value")
        if pd is None:
            raise QuickStatsError("pandas is not installed. Try `pip install pandas`.")
        return pd.DataFrame(data)

    async def fetch_all_async(
        self,
        *,
        year_batch_size: int = 5,
        sleep: float = 0.0,
        max_concurrent: int = 3,
        **filters: Any,
    ) -> List[Dict[str, Any]]:
        """Async version of fetch_all() with concurrency control for faster multi-year requests.
        
        Args:
            year_batch_size: Years per batch (default: 5)
            sleep: Sleep between requests (default: 0.0)
            max_concurrent: Maximum concurrent requests (default: 3)
            **filters: Query parameters
            
        Returns:
            List of record dictionaries
        """
        # Build a candidate list of years based on filters
        years = self._candidate_years(**filters)
        if not years:
            # Fallback: try without scoping
            years = sorted(set(self.get_param_values("year")))
        
        results: List[Dict[str, Any]] = []
        # Remove any explicit year filters when chunking; we will add them per batch.
        scope_filters = {k: v for k, v in filters.items() if not k.startswith("year")}

        def batches(seq: List[int], k: int) -> Iterable[List[int]]:
            for i in range(0, len(seq), k):
                yield seq[i:i+k]

        # Create semaphore for concurrency control
        semaphore = asyncio.Semaphore(max_concurrent)
        
        async def process_chunk(chunk: List[int]) -> List[Dict[str, Any]]:
            """Process a single chunk of years with concurrency control."""
            async with semaphore:
                try:
                    # Adaptively split until count <= limit
                    lo, hi = chunk[0], chunk[-1]
                    subfilters = {**scope_filters}
                    if lo == hi:
                        subfilters["year"] = lo
                    else:
                        subfilters["year__GE"] = lo
                        subfilters["year__LE"] = hi
                    
                    count = self.get_counts(**subfilters)
                    if count == 0:
                        return []
                    
                    # If too big, split the chunk
                    if count > self.HARD_ROW_LIMIT and len(chunk) > 1:
                        # Split recursively by halving the year range until under the limit
                        chunk_results = []
                        for sub in self._split_until_under_limit(chunk, scope_filters):
                            chunk_data = await self.api_get_async(sub, fmt="JSON")
                            chunk_results.extend(chunk_data)
                            if sleep:
                                await asyncio.sleep(sleep)
                        return chunk_results
                    else:
                        chunk_data = await self.api_get_async(subfilters, fmt="JSON")
                        if sleep:
                            await asyncio.sleep(sleep)
                        return chunk_data
                        
                except Exception as e:
                    print(f"Error processing chunk {chunk}: {e}")
                    return []

        # Create tasks for all chunks
        tasks = [process_chunk(chunk) for chunk in batches(years, year_batch_size)]
        
        # Execute all tasks concurrently
        chunk_results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Collect results from all chunks
        for chunk_result in chunk_results:
            if isinstance(chunk_result, Exception):
                print(f"Chunk failed with error: {chunk_result}")
                continue
            if isinstance(chunk_result, list):
                results.extend(chunk_result)
        
        return results

    async def api_get_async(self, params: Dict[str, Any], fmt: str = "JSON") -> Union[List[Dict[str, Any]], str]:
        """Async version of api_get() for concurrent requests."""
        fmt_upper = fmt.upper()
        if fmt_upper not in {"JSON", "CSV", "XML"}:
            raise QuickStatsError("fmt must be one of JSON, CSV, XML")
        
        payload = self._with_key(params)
        payload["format"] = fmt_upper
        url = f"{self.base_url}/api_GET/"
        
        async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=self.timeout)) as session:
            resp = await self._request_async(session, "GET", url, params=payload)
            
            if fmt_upper == "JSON":
                try:
                    obj = await resp.json()
                    if "error" in obj:
                        raise QuickStatsError(f"/api_GET error: {obj['error']}")
                    data = obj.get("data")
                    if not isinstance(data, list):
                        raise QuickStatsError(f"Unexpected /api_GET JSON: {obj}")
                    # Normalize keys: some responses use 'Value', some 'Value ' - be tolerant
                    normed = []
                    for rec in data:
                        if not isinstance(rec, dict):
                            continue
                        # Strip whitespace from keys
                        normed.append({str(k).strip(): v for k, v in rec.items()})
                    return normed
                except ValueError as e:
                    response_text = await resp.text()
                    raise QuickStatsError(f"Invalid /api_GET JSON: {response_text[:400]}") from e
            else:
                return await resp.text()

    async def _request_async(self, session: aiohttp.ClientSession, method: str, url: str, *, params: Optional[Dict[str, Any]] = None) -> aiohttp.ClientResponse:
        """Async version of _request() with retry logic."""
        last_exc: Optional[Exception] = None
        
        for attempt in range(1, self.max_retries + 2):
            try:
                async with session.request(method, url, params=params) as resp:
                    # Basic retry on 5xx
                    if resp.status >= 500 and attempt <= self.max_retries:
                        await asyncio.sleep(self.backoff * attempt)
                        continue
                    
                    # Raise for HTTP errors
                    if resp.status >= 400:
                        # Try to show API-provided error body
                        message = await resp.text()
                        message = message.strip()
                        raise QuickStatsError(f"HTTP {resp.status} for {url}: {message[:400]}")
                    
                    return resp
                    
            except (aiohttp.ClientError, asyncio.TimeoutError) as e:
                last_exc = e
                if attempt <= self.max_retries:
                    await asyncio.sleep(self.backoff * attempt)
                    continue
                raise QuickStatsError(f"Network error contacting Quick Stats API: {e}") from e
        
        # Should not reach here
        assert False, f"Unreachable; last_exc={last_exc}"

if __name__ == "__main__":  # Simple smoke test (requires an API key in env)
    try:
        client = QuickStatsClient()
        c = client.get_counts(commodity_desc="CORN",agg_level_desc="COUNTY",short_desc="CORN - ACRES PLANTED", year__GE=2010)
        print("Corn IA since 2019 count:", c)
        if c and c < QuickStatsClient.HARD_ROW_LIMIT:
            data = client.query(commodity_desc="CORN", state_alpha="IA", year__GE=2019, agg_level_desc="STATE")
            print("Sample record keys:", list(data[0].keys()) if data else [])
    except QuickStatsError as e:
        print("QuickStatsError:", e)
    except Exception as e:
        print("Unhandled error:", e)
