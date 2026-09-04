"""
Automated dataset profiling for InsightAI Analytics.

This module analyzes the structure and basic characteristics
of a pandas DataFrame.
"""

from typing import Any

import numpy as np
import pandas as pd


def profile_dataset(df: pd.DataFrame) -> dict[str, Any]:
    """
    Generate a comprehensive profile of a DataFrame.

    Parameters
    ----------
    df:
        Input pandas DataFrame.

    Returns
    -------
    dict
        Structured dataset profile.
    """

    if df is None:
        raise ValueError("DataFrame cannot be None.")

    if df.empty:
        raise ValueError("DataFrame is empty.")

    numeric_columns = df.select_dtypes(
        include=np.number
    ).columns.tolist()

    categorical_columns = df.select_dtypes(
        include=["object", "category", "bool"]
    ).columns.tolist()

    datetime_columns = df.select_dtypes(
        include=["datetime", "datetimetz"]
    ).columns.tolist()

    profile = {
        "rows": int(df.shape[0]),
        "columns": int(df.shape[1]),
        "total_cells": int(df.shape[0] * df.shape[1]),
        "numeric_columns": numeric_columns,
        "categorical_columns": categorical_columns,
        "datetime_columns": datetime_columns,
        "column_names": df.columns.tolist(),
        "duplicate_rows": int(df.duplicated().sum()),
        "memory_usage_mb": round(
            df.memory_usage(deep=True).sum() / (1024 ** 2),
            2,
        ),
    }

    profile["missing_values"] = get_missing_value_summary(df)
    profile["column_summary"] = get_column_summary(df)

    return profile


def get_missing_value_summary(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Calculate missing-value statistics for every column.
    """

    missing_count = df.isna().sum()

    missing_percentage = (
        missing_count / len(df) * 100
    ).round(2)

    summary = pd.DataFrame(
        {
            "missing_count": missing_count,
            "missing_percentage": missing_percentage,
        }
    )

    summary["complete_count"] = (
        len(df) - summary["missing_count"]
    )

    summary = summary.sort_values(
        by="missing_percentage",
        ascending=False,
    )

    return summary


def get_column_summary(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Generate column-level metadata.
    """

    records = []

    for column in df.columns:

        series = df[column]

        records.append(
            {
                "column": column,
                "dtype": str(series.dtype),
                "non_null": int(series.notna().sum()),
                "missing": int(series.isna().sum()),
                "missing_pct": round(
                    series.isna().mean() * 100,
                    2,
                ),
                "unique_values": int(
                    series.nunique(dropna=True)
                ),
            }
        )

    return pd.DataFrame(records)


def get_numeric_summary(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Generate descriptive statistics for numerical columns.
    """

    numeric_df = df.select_dtypes(
        include=np.number
    )

    if numeric_df.empty:
        return pd.DataFrame()

    summary = numeric_df.describe().T

    summary["median"] = numeric_df.median()

    summary["missing"] = numeric_df.isna().sum()

    summary["missing_pct"] = (
        numeric_df.isna().mean() * 100
    ).round(2)

    return summary


def get_categorical_summary(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Generate summary information for categorical columns.
    """

    categorical_columns = df.select_dtypes(
        include=["object", "category", "bool"]
    ).columns

    records = []

    for column in categorical_columns:

        series = df[column]

        mode_values = series.mode(dropna=True)

        most_common = (
            mode_values.iloc[0]
            if not mode_values.empty
            else None
        )

        records.append(
            {
                "column": column,
                "unique_values": int(
                    series.nunique(dropna=True)
                ),
                "missing": int(series.isna().sum()),
                "missing_pct": round(
                    series.isna().mean() * 100,
                    2,
                ),
                "most_common": most_common,
                "most_common_count": (
                    int(series.value_counts().iloc[0])
                    if not series.dropna().empty
                    else 0
                ),
            }
        )

    return pd.DataFrame(records)


def detect_possible_date_columns(
    df: pd.DataFrame,
) -> list[str]:
    """
    Detect columns that may contain date/time information.

    Existing datetime columns are included automatically.
    Object/string columns are tested conservatively.
    """

    date_columns = []

    for column in df.columns:

        series = df[column]

        if pd.api.types.is_datetime64_any_dtype(series):
            date_columns.append(column)
            continue

        if not pd.api.types.is_object_dtype(series):
            continue

        sample = series.dropna()

        if sample.empty:
            continue

        sample = sample.head(min(500, len(sample)))

        converted = pd.to_datetime(
            sample,
            errors="coerce",
        )

        conversion_rate = converted.notna().mean()

        if conversion_rate >= 0.8:
            date_columns.append(column)

    return date_columns

