"""
Data quality analysis for InsightAI Analytics.

This module identifies common data-quality problems before
visualization, statistical analysis, or machine learning.
"""

from typing import Any

import numpy as np
import pandas as pd


def analyze_data_quality(df: pd.DataFrame) -> dict[str, Any]:
    """
    Run a complete data-quality assessment.

    Parameters
    ----------
    df:
        Input pandas DataFrame.

    Returns
    -------
    dict
        Data-quality results.
    """

    if df is None:
        raise ValueError("DataFrame cannot be None.")

    if df.empty:
        raise ValueError("DataFrame is empty.")

    missing = analyze_missing_values(df)
    duplicates = analyze_duplicates(df)
    constants = detect_constant_columns(df)
    near_constants = detect_near_constant_columns(df)
    high_cardinality = detect_high_cardinality_columns(df)
    potential_ids = detect_potential_id_columns(df)
    outliers = detect_numeric_outliers(df)

    quality_score = calculate_quality_score(
        df=df,
        missing=missing,
        duplicate_count=duplicates["duplicate_count"],
        outlier_summary=outliers,
    )

    return {
        "quality_score": quality_score,
        "missing": missing,
        "duplicates": duplicates,
        "constant_columns": constants,
        "near_constant_columns": near_constants,
        "high_cardinality_columns": high_cardinality,
        "potential_id_columns": potential_ids,
        "outliers": outliers,
    }


def analyze_missing_values(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Analyze missing values by column.
    """

    result = pd.DataFrame(
        {
            "column": df.columns,
            "missing_count": df.isna().sum().values,
            "missing_percentage": (
                df.isna().mean().values * 100
            ).round(2),
        }
    )

    result["severity"] = result[
        "missing_percentage"
    ].apply(classify_missing_severity)

    return result.sort_values(
        "missing_percentage",
        ascending=False,
    ).reset_index(drop=True)


def classify_missing_severity(
    percentage: float,
) -> str:
    """
    Classify missing-value severity.

    Thresholds are intentionally simple and transparent.
    """

    if percentage == 0:
        return "None"

    if percentage < 5:
        return "Low"

    if percentage < 20:
        return "Moderate"

    if percentage < 50:
        return "High"

    return "Critical"


def analyze_duplicates(
    df: pd.DataFrame,
) -> dict[str, Any]:
    """
    Analyze duplicate records.
    """

    duplicate_count = int(df.duplicated().sum())

    total_rows = len(df)

    duplicate_percentage = (
        duplicate_count / total_rows * 100
        if total_rows > 0
        else 0
    )

    return {
        "duplicate_count": duplicate_count,
        "duplicate_percentage": round(
            duplicate_percentage,
            2,
        ),
    }


def detect_constant_columns(
    df: pd.DataFrame,
) -> list[str]:
    """
    Detect columns containing only one unique value.
    """

    constant_columns = []

    for column in df.columns:

        unique_count = df[column].nunique(
            dropna=False
        )

        if unique_count <= 1:
            constant_columns.append(column)

    return constant_columns


def detect_near_constant_columns(
    df: pd.DataFrame,
    threshold: float = 0.99,
) -> list[dict[str, Any]]:
    """
    Detect columns where one value dominates most records.

    Example:
        If 99.5% of rows contain "No", the column may
        provide very little predictive information.
    """

    results = []

    for column in df.columns:

        value_counts = df[column].value_counts(
            normalize=True,
            dropna=False,
        )

        if value_counts.empty:
            continue

        dominant_percentage = (
            value_counts.iloc[0] * 100
        )

        if value_counts.iloc[0] >= threshold:
            results.append(
                {
                    "column": column,
                    "dominant_percentage": round(
                        dominant_percentage,
                        2,
                    ),
                }
            )

    return results


def detect_high_cardinality_columns(
    df: pd.DataFrame,
    threshold: float = 0.5,
) -> list[dict[str, Any]]:
    """
    Detect categorical/text columns with unusually high
    numbers of unique values.

    A column is considered high-cardinality when its
    unique-value ratio is at least the given threshold.
    """

    results = []

    row_count = len(df)

    if row_count == 0:
        return results

    categorical_columns = df.select_dtypes(
        include=["object", "category"]
    ).columns

    for column in categorical_columns:

        unique_count = df[column].nunique(
            dropna=True
        )

        unique_ratio = unique_count / row_count

        if unique_ratio >= threshold:

            results.append(
                {
                    "column": column,
                    "unique_values": int(unique_count),
                    "unique_ratio": round(
                        unique_ratio * 100,
                        2,
                    ),
                }
            )

    return results


def detect_potential_id_columns(
    df: pd.DataFrame,
) -> list[str]:
    """
    Detect columns that may represent identifiers.

    This is a heuristic, not a guarantee.
    """

    potential_ids = []

    row_count = len(df)

    if row_count == 0:
        return potential_ids

    id_keywords = {
        "id",
        "identifier",
        "customer_id",
        "customerid",
        "user_id",
        "userid",
        "transaction_id",
        "transactionid",
        "order_id",
        "orderid",
        "account_id",
        "accountid",
    }

    for column in df.columns:

        normalized = (
            str(column)
            .strip()
            .lower()
            .replace("-", "_")
            .replace(" ", "_")
        )

        unique_count = df[column].nunique(
            dropna=True
        )

        unique_ratio = unique_count / row_count

        keyword_match = (
            normalized in id_keywords
            or normalized.endswith("_id")
            or normalized.endswith("id")
        )

        high_uniqueness = unique_ratio >= 0.95

        if keyword_match or high_uniqueness:
            potential_ids.append(column)

    return potential_ids


def detect_numeric_outliers(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Detect numeric outliers using the IQR method.

    Outlier detection is performed independently for
    each numerical column.
    """

    numeric_columns = df.select_dtypes(
        include=np.number
    ).columns

    results = []

    for column in numeric_columns:

        series = df[column].dropna()

        if series.empty:
            continue

        q1 = series.quantile(0.25)
        q3 = series.quantile(0.75)

        iqr = q3 - q1

        if iqr == 0:
            outlier_count = 0

        else:
            lower_bound = q1 - 1.5 * iqr
            upper_bound = q3 + 1.5 * iqr

            outlier_count = int(
                (
                    (series < lower_bound)
                    | (series > upper_bound)
                ).sum()
            )

        outlier_percentage = (
            outlier_count / len(series) * 100
        )

        results.append(
            {
                "column": column,
                "outlier_count": outlier_count,
                "outlier_percentage": round(
                    outlier_percentage,
                    2,
                ),
            }
        )

    return pd.DataFrame(results).sort_values(
        "outlier_percentage",
        ascending=False,
    ).reset_index(drop=True)


def calculate_quality_score(
    df: pd.DataFrame,
    missing: pd.DataFrame,
    duplicate_count: int,
    outlier_summary: pd.DataFrame,
) -> float:
    """
    Calculate a simple 0-100 data-quality score.

    The score is a screening metric, not a statistical
    measure of dataset correctness.
    """

    score = 100.0

    total_cells = df.shape[0] * df.shape[1]

    if total_cells > 0:

        missing_cells = int(
            df.isna().sum().sum()
        )

        missing_rate = (
            missing_cells / total_cells
        )

        score -= min(
            missing_rate * 40,
            40,
        )

    if len(df) > 0:

        duplicate_rate = (
            duplicate_count / len(df)
        )

        score -= min(
            duplicate_rate * 20,
            20,
        )

    constant_count = len(
        detect_constant_columns(df)
    )

    if df.shape[1] > 0:

        constant_rate = (
            constant_count / df.shape[1]
        )

        score -= min(
            constant_rate * 20,
            20,
        )

    if not outlier_summary.empty:

        average_outlier_rate = (
            outlier_summary["outlier_percentage"]
            .mean()
        )

        score -= min(
            average_outlier_rate * 0.2,
            10,
        )

    return round(
        max(score, 0),
        2,
    )
