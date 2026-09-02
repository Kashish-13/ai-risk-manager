"""Loading and target extraction for tabular fraud datasets."""
from __future__ import annotations

from pathlib import Path

import pandas as pd


def load_csv(path: Path | str, *, require_nonempty: bool = True) -> pd.DataFrame:
    """Read a CSV and raise a useful error for common input problems."""
    file_path = Path(path)
    if not file_path.exists():
        raise FileNotFoundError(f"Dataset not found: {file_path}")
    if file_path.suffix.lower() != ".csv":
        raise ValueError(f"Expected a .csv file, received: {file_path.name}")
    try:
        frame = pd.read_csv(file_path)
    except (OSError, UnicodeDecodeError, pd.errors.ParserError) as exc:
        raise ValueError(f"Could not parse CSV '{file_path}': {exc}") from exc
    if require_nonempty and frame.empty:
        raise ValueError(f"Dataset '{file_path}' contains no rows.")
    if frame.columns.duplicated().any():
        duplicates = frame.columns[frame.columns.duplicated()].tolist()
        raise ValueError(f"Dataset contains duplicate column names: {duplicates}")
    return frame


def require_target(frame: pd.DataFrame, target_column: str) -> None:
    """Confirm the target is present and usable without changing the frame."""
    if target_column not in frame.columns:
        raise ValueError(
            f"Target column '{target_column}' is missing. Available columns: {frame.columns.tolist()}"
        )
    if frame[target_column].isna().any():
        raise ValueError(f"Target column '{target_column}' contains missing values.")


def split_features_target(frame: pd.DataFrame, target_column: str) -> tuple[pd.DataFrame, pd.Series]:
    """Return copies of features and labels, preserving the source dataframe."""
    require_target(frame, target_column)
    return frame.drop(columns=[target_column]).copy(), frame[target_column].copy()

