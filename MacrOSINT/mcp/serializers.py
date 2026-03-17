import pandas as pd
import json


def format_df_for_llm(df: pd.DataFrame, max_rows: int = 100, summarize: bool = False) -> str:
    """Safely formats large DataFrames for the LLM context window."""
    if df is None or df.empty:
        return "No data found for the requested parameters."

    if summarize:
        return df.describe().to_json(orient="columns")

    if len(df) > max_rows:
        df = df.tail(max_rows)

    if isinstance(df.index, pd.DatetimeIndex):
        df = df.reset_index()

    # Flatten MultiIndex columns for JSON serialization
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [" | ".join(str(c) for c in col if c) for col in df.columns]

    return df.to_json(orient="records", date_format="iso")
