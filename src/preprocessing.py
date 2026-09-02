"""Leakage-safe sklearn preprocessing and split utilities."""
from __future__ import annotations

from dataclasses import dataclass

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import StratifiedGroupKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


@dataclass(frozen=True)
class DatasetSplits:
    X_train: pd.DataFrame
    X_validation: pd.DataFrame
    X_test: pd.DataFrame
    y_train: pd.Series
    y_validation: pd.Series
    y_test: pd.Series


def split_train_validation_test(
    X: pd.DataFrame,
    y: pd.Series,
    *,
    test_size: float,
    validation_size: float,
    random_seed: int,
    stratify: bool = True,
    groups: pd.Series | None = None,
) -> DatasetSplits:
    """Create reproducible stratified splits, reserving test data from selection."""
    if not 0 < test_size < 1 or not 0 < validation_size < 1:
        raise ValueError("test_size and validation_size must each be between 0 and 1.")
    if test_size + validation_size >= 1:
        raise ValueError("test_size + validation_size must be less than 1.")
    if len(X) != len(y):
        raise ValueError("X and y must contain the same number of rows.")
    if groups is not None:
        if len(groups) != len(X):
            raise ValueError("groups must contain one value per feature row.")
        if not stratify:
            raise ValueError("Grouped splitting currently requires stratify=True.")
        # Two deterministic stratified group splits yield 60/20/20 without an
        # identical transaction-feature group crossing any split boundary.
        outer = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=random_seed)
        remaining_positions, test_positions = next(outer.split(X, y, groups))
        X_remainder, X_test = X.iloc[remaining_positions], X.iloc[test_positions]
        y_remainder, y_test = y.iloc[remaining_positions], y.iloc[test_positions]
        remaining_groups = groups.iloc[remaining_positions]
        inner = StratifiedGroupKFold(n_splits=4, shuffle=True, random_state=random_seed)
        train_relative, validation_relative = next(inner.split(X_remainder, y_remainder, remaining_groups))
        return DatasetSplits(
            X_remainder.iloc[train_relative], X_remainder.iloc[validation_relative], X_test,
            y_remainder.iloc[train_relative], y_remainder.iloc[validation_relative], y_test,
        )
    stratification = y if stratify else None
    try:
        X_remainder, X_test, y_remainder, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_seed, stratify=stratification
        )
        validation_share_of_remainder = validation_size / (1 - test_size)
        remainder_stratification = y_remainder if stratify else None
        X_train, X_validation, y_train, y_validation = train_test_split(
            X_remainder, y_remainder, test_size=validation_share_of_remainder,
            random_state=random_seed, stratify=remainder_stratification,
        )
    except ValueError as exc:
        raise ValueError(
            "Could not create stratified splits. Each class needs enough examples for all splits. "
            "For tiny development data, disable stratification explicitly."
        ) from exc
    return DatasetSplits(X_train, X_validation, X_test, y_train, y_validation, y_test)


def build_preprocessor(X_train: pd.DataFrame) -> ColumnTransformer:
    """Build an unfitted transformer; call fit only with training features."""
    numeric_columns = X_train.select_dtypes(include="number").columns.tolist()
    categorical_columns = [column for column in X_train.columns if column not in numeric_columns]
    transformers = []
    if numeric_columns:
        transformers.append(("numeric", Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]), numeric_columns))
    if categorical_columns:
        transformers.append(("categorical", Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore")),
        ]), categorical_columns))
    if not transformers:
        raise ValueError("Cannot build a preprocessor with no feature columns.")
    return ColumnTransformer(transformers=transformers, remainder="drop")
