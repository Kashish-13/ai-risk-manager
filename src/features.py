"""Dataset-aware, deterministic transaction feature engineering."""
from __future__ import annotations

import numpy as np
import pandas as pd


def _add_time_features(result: pd.DataFrame, column: str) -> None:
    numeric_time = pd.to_numeric(result[column], errors="coerce")
    # ULB's Time is elapsed seconds; these are cyclic clock-time proxies, not actual timestamps.
    hour = (numeric_time % 86_400) / 3_600
    result[f"{column}_hour"] = hour
    result[f"{column}_hour_sin"] = np.sin(2 * np.pi * hour / 24)
    result[f"{column}_hour_cos"] = np.cos(2 * np.pi * hour / 24)
    result[f"{column}_day_index"] = np.floor(numeric_time / 86_400)


def _add_datetime_features(result: pd.DataFrame, column: str) -> bool:
    parsed = pd.to_datetime(result[column], errors="coerce", utc=True)
    if parsed.notna().sum() == 0:
        return False
    result[f"{column}_hour"] = parsed.dt.hour.astype("float64")
    result[f"{column}_day_of_week"] = parsed.dt.dayofweek.astype("float64")
    return True


def engineer_features(
    frame: pd.DataFrame,
    *,
    time_column: str = "Time",
    amount_column: str = "Amount",
) -> pd.DataFrame:
    """Add only deterministic features justified by columns that actually exist.

    The original columns are retained. No label-derived, group-history, or fitted
    statistics are constructed here, so this function is safe before splitting.
    """
    result = frame.copy()
    if time_column in result.columns:
        if pd.api.types.is_numeric_dtype(result[time_column]):
            _add_time_features(result, time_column)
        else:
            _add_datetime_features(result, time_column)
    if amount_column in result.columns:
        amount = pd.to_numeric(result[amount_column], errors="coerce")
        # Transaction values cannot be negative in the expected schema; clip only
        # for the derived transform and preserve the original input unchanged.
        nonnegative_amount = amount.clip(lower=0)
        result[f"{amount_column}_log1p"] = np.log1p(nonnegative_amount)
        result[f"{amount_column}_sqrt"] = np.sqrt(nonnegative_amount)
        result[f"{amount_column}_is_zero"] = (amount == 0).astype("int8")
        result[f"{amount_column}_squared"] = nonnegative_amount.pow(2)
    return result

