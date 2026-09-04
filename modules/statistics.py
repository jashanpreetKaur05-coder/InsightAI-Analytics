"""
Statistical analysis engine for InsightAI Analytics.

Provides reusable statistical calculations that can be
used by the dashboard and AI analyst.
"""

from typing import Any, Optional

import numpy as np
import pandas as pd
from scipy import stats


def numeric_summary(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Generate descriptive statistics for numerical columns.
    """

    numeric_df = df.select_dtypes(include=np.number)

    if numeric_df.empty:
        return pd.DataFrame()

    summary = numeric_df.describe().T

    summary["median"] = numeric_df.median()
    summary["variance"] = numeric_df.var()
    summary["skewness"] = numeric_df.skew()
    summary["kurtosis"] = numeric_df.kurtosis()

    return summary.reset_index().rename(
        columns={"index": "column"}
    )


def correlation_matrix(
    df: pd.DataFrame,
    method: str = "pearson",
) -> pd.DataFrame:
    """
    Calculate correlation between numerical variables.

    Supported methods:
        pearson
        spearman
        kendall
    """

    numeric_df = df.select_dtypes(
        include=np.number
    )

    if numeric_df.shape[1] < 2:
        return pd.DataFrame()

    if method not in {
        "pearson",
        "spearman",
        "kendall",
    }:
        raise ValueError(
            "method must be pearson, spearman, or kendall."
        )

    return numeric_df.corr(method=method)


def strongest_correlations(
    df: pd.DataFrame,
    top_n: int = 10,
    method: str = "pearson",
) -> pd.DataFrame:
    """
    Find the strongest numerical relationships.

    Self-correlations and duplicate mirrored pairs are removed.
    """

    matrix = correlation_matrix(
        df,
        method=method,
    )

    if matrix.empty:
        return pd.DataFrame()

    records = []

    columns = matrix.columns.tolist()

    for i in range(len(columns)):

        for j in range(i + 1, len(columns)):

            column_a = columns[i]
            column_b = columns[j]

            coefficient = matrix.loc[
                column_a,
                column_b,
            ]

            if pd.isna(coefficient):
                continue

            records.append(
                {
                    "variable_1": column_a,
                    "variable_2": column_b,
                    "correlation": round(
                        float(coefficient),
                        4,
                    ),
                    "absolute_correlation": round(
                        abs(float(coefficient)),
                        4,
                    ),
                    "relationship_strength": (
                        classify_correlation(
                            float(coefficient)
                        )
                    ),
                }
            )

    result = pd.DataFrame(records)

    if result.empty:
        return result

    return (
        result
        .sort_values(
            "absolute_correlation",
            ascending=False,
        )
        .head(top_n)
        .reset_index(drop=True)
    )


def classify_correlation(
    coefficient: float,
) -> str:
    """
    Classify the strength of a correlation.

    This describes association strength only.
    It does not imply causation.
    """

    absolute_value = abs(coefficient)

    if absolute_value < 0.2:
        return "Very weak"

    if absolute_value < 0.4:
        return "Weak"

    if absolute_value < 0.6:
        return "Moderate"

    if absolute_value < 0.8:
        return "Strong"

    return "Very strong"


def correlation_p_value(
    df: pd.DataFrame,
    column_a: str,
    column_b: str,
) -> Optional[dict[str, float]]:
    """
    Calculate Pearson correlation and its p-value
    between two numerical variables.
    """

    if column_a not in df.columns:
        raise ValueError(
            f"Column '{column_a}' does not exist."
        )

    if column_b not in df.columns:
        raise ValueError(
            f"Column '{column_b}' does not exist."
        )

    data = df[
        [column_a, column_b]
    ].dropna()

    if len(data) < 3:
        return None

    if (
        not pd.api.types.is_numeric_dtype(data[column_a])
        or not pd.api.types.is_numeric_dtype(data[column_b])
    ):
        raise ValueError(
            "Both columns must be numerical."
        )

    coefficient, p_value = stats.pearsonr(
        data[column_a],
        data[column_b],
    )

    return {
        "correlation": round(
            float(coefficient),
            6,
        ),
        "p_value": round(
            float(p_value),
            6,
        ),
        "sample_size": int(len(data)),
    }


def normality_test(
    df: pd.DataFrame,
    column: str,
) -> Optional[dict[str, Any]]:
    """
    Perform a Shapiro-Wilk normality test.

    For very large datasets, a sample of 5,000 observations
    is used to keep the test computationally manageable.
    """

    if column not in df.columns:
        raise ValueError(
            f"Column '{column}' does not exist."
        )

    if not pd.api.types.is_numeric_dtype(df[column]):
        raise ValueError(
            "Normality testing requires a numerical column."
        )

    values = df[column].dropna()

    if len(values) < 3:
        return None

    if len(values) > 5000:
        values = values.sample(
            n=5000,
            random_state=42,
        )

    statistic, p_value = stats.shapiro(values)

    return {
        "column": column,
        "test": "Shapiro-Wilk",
        "statistic": round(
            float(statistic),
            6,
        ),
        "p_value": round(
            float(p_value),
            6,
        ),
        "sample_size": int(len(values)),
        "interpretation": (
            "Evidence against normality"
            if p_value < 0.05
            else "No strong evidence against normality"
        ),
    }


def group_statistics(
    df: pd.DataFrame,
    category_column: str,
    value_column: str,
) -> pd.DataFrame:
    """
    Calculate business-friendly statistics by category.
    """

    if category_column not in df.columns:
        raise ValueError(
            f"Column '{category_column}' does not exist."
        )

    if value_column not in df.columns:
        raise ValueError(
            f"Column '{value_column}' does not exist."
        )

    if not pd.api.types.is_numeric_dtype(
        df[value_column]
    ):
        raise ValueError(
            f"'{value_column}' must be numerical."
        )

    result = (
        df.groupby(
            category_column,
            dropna=False,
        )[value_column]
        .agg(
            [
                "count",
                "sum",
                "mean",
                "median",
                "min",
                "max",
                "std",
            ]
        )
        .reset_index()
    )

    return result.sort_values(
        "sum",
        ascending=False,
    ).reset_index(drop=True)


def compare_two_groups(
    df: pd.DataFrame,
    category_column: str,
    value_column: str,
) -> Optional[dict[str, Any]]:
    """
    Compare exactly two groups using Welch's t-test.

    This is appropriate when comparing the means of two
    independent groups, subject to the usual assumptions.
    """

    if category_column not in df.columns:
        raise ValueError(
            f"Column '{category_column}' does not exist."
        )

    if value_column not in df.columns:
        raise ValueError(
            f"Column '{value_column}' does not exist."
        )

    data = df[
        [category_column, value_column]
    ].dropna()

    groups = data[category_column].unique()

    if len(groups) != 2:
        return None

    group_a = data.loc[
        data[category_column] == groups[0],
        value_column,
    ]

    group_b = data.loc[
        data[category_column] == groups[1],
        value_column,
    ]

    if len(group_a) < 2 or len(group_b) < 2:
        return None

    statistic, p_value = stats.ttest_ind(
        group_a,
        group_b,
        equal_var=False,
    )

    return {
        "group_1": str(groups[0]),
        "group_2": str(groups[1]),
        "group_1_mean": round(
            float(group_a.mean()),
            4,
        ),
        "group_2_mean": round(
            float(group_b.mean()),
            4,
        ),
        "mean_difference": round(
            float(group_a.mean() - group_b.mean()),
            4,
        ),
        "t_statistic": round(
            float(statistic),
            6,
        ),
        "p_value": round(
            float(p_value),
            6,
        ),
        "statistically_significant_at_05": (
            bool(p_value < 0.05)
        ),
    }


def frequency_table(
    df: pd.DataFrame,
    column: str,
    top_n: int = 20,
) -> pd.DataFrame:
    """
    Create a frequency table for a categorical column.
    """

    if column not in df.columns:
        raise ValueError(
            f"Column '{column}' does not exist."
        )

    result = (
        df[column]
        .value_counts(
            dropna=False,
        )
        .head(top_n)
        .reset_index()
    )

    result.columns = [
        column,
        "count",
    ]

    result["percentage"] = (
        result["count"] / len(df) * 100
    ).round(2)

    return result


def detect_numeric_skewness(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Identify skewness in numerical variables.
    """

    numeric_df = df.select_dtypes(
        include=np.number
    )

    if numeric_df.empty:
        return pd.DataFrame()

    records = []

    for column in numeric_df.columns:

        values = numeric_df[column].dropna()

        if len(values) < 3:
            continue

        skewness = float(values.skew())

        if abs(skewness) < 0.5:
            interpretation = "Approximately symmetric"

        elif abs(skewness) < 1:
            interpretation = "Moderately skewed"

        else:
            interpretation = "Highly skewed"

        records.append(
            {
                "column": column,
                "skewness": round(
                    skewness,
                    4,
                ),
                "interpretation": interpretation,
            }
        )

    return (
        pd.DataFrame(records)
        .sort_values(
            "skewness",
            key=lambda x: x.abs(),
            ascending=False,
        )
        .reset_index(drop=True)
    )
