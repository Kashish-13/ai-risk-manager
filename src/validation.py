"""Non-mutating schema and quality checks for tabular datasets."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class ValidationReport:
    rows: int
    columns: int
    missing_values: dict[str, int]
    duplicate_rows: int
    target_distribution: dict[object, int]
    numeric_columns: list[str]
    categorical_columns: list[str]
    invalid_numeric_values: dict[str, int]
    errors: list[str]
    warnings: list[str]

    @property
    def is_valid(self) -> bool:
        return not self.errors


def validate_dataframe(frame: pd.DataFrame, target_column: str) -> ValidationReport:
    """Inspect a dataframe without imputing, dropping, or coercing any values."""
    errors: list[str] = []
    warnings: list[str] = []
    if frame.empty:
        errors.append("Dataset has no rows.")
    if frame.columns.duplicated().any():
        errors.append("Dataset has duplicate column names.")
    if target_column not in frame.columns:
        errors.append(f"Target column '{target_column}' is missing.")
        distribution: dict[object, int] = {}
    else:
        target = frame[target_column]
        distribution = target.value_counts(dropna=False).to_dict()
        if target.isna().any():
            errors.append(f"Target column '{target_column}' contains missing values.")
        if target.nunique(dropna=True) < 2:
            errors.append("Target must contain at least two classes.")
    numeric = frame.select_dtypes(include=[np.number]).columns.tolist()
    categorical = [column for column in frame.columns if column not in numeric]
    invalid: dict[str, int] = {}
    for column in numeric:
        values = frame[column].to_numpy(dtype=float, na_value=np.nan)
        count = int(np.isinf(values).sum())
        if count:
            invalid[column] = count
    if invalid:
        errors.append("Numeric columns contain infinite values.")
    missing = {column: int(count) for column, count in frame.isna().sum().items() if count}
    if missing:
        warnings.append("Missing values detected; preprocessing must impute them using training data only.")
    duplicates = int(frame.duplicated().sum())
    if duplicates:
        warnings.append("Duplicate rows detected; no rows were removed during validation.")
    return ValidationReport(
        rows=len(frame), columns=len(frame.columns), missing_values=missing,
        duplicate_rows=duplicates, target_distribution=distribution,
        numeric_columns=numeric, categorical_columns=categorical,
        invalid_numeric_values=invalid, errors=errors, warnings=warnings,
    )

