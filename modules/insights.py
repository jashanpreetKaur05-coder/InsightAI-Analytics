"""
Automated business insights engine for InsightAI Analytics.

This module converts statistical and data-quality results
into transparent, rule-based observations.

The generated insights are evidence-based and do not claim
causation where the data only shows association.
"""

from typing import Any

import numpy as np
import pandas as pd


def generate_insights(
    df: pd.DataFrame,
    max_insights: int = 15,
) -> list[dict[str, Any]]:
    """
    Generate automated insights from a dataset.

    Each insight contains:
        category
        title
        message
        severity
        evidence
    """

    if df is None:
        raise ValueError("DataFrame cannot be None.")

    if df.empty:
        return []

    insights: list[dict[str, Any]] = []

    insights.extend(
        _missing_value_insights(df)
    )

    insights.extend(
        _duplicate_insights(df)
    )

    insights.extend(
        _numeric_distribution_insights(df)
    )

    insights.extend(
        _correlation_insights(df)
    )

    insights.extend(
        _categorical_insights(df)
    )

    insights.extend(
        _potential_id_insights(df)
    )

    insights = _sort_insights(insights)

    return insights[:max_insights]


def _missing_value_insights(
    df: pd.DataFrame,
) -> list[dict[str, Any]]:
    """Generate insights about missing data."""

    results = []

    total_rows = len(df)

    if total_rows == 0:
        return results

    for column in df.columns:

        missing_count = int(
            df[column].isna().sum()
        )

        if missing_count == 0:
            continue

        percentage = (
            missing_count / total_rows * 100
        )

        if percentage >= 50:
            severity = "critical"

        elif percentage >= 20:
            severity = "high"

        elif percentage >= 5:
            severity = "medium"

        else:
            severity = "low"

        results.append(
            {
                "category": "Data Quality",
                "title": f"Missing values in {column}",
                "message": (
                    f"{column} contains "
                    f"{missing_count:,} missing values "
                    f"({percentage:.1f}% of rows)."
                ),
                "severity": severity,
                "evidence": {
                    "column": column,
                    "missing_count": missing_count,
                    "missing_percentage": round(
                        percentage,
                        2,
                    ),
                },
            }
        )

    return results


def _duplicate_insights(
    df: pd.DataFrame,
) -> list[dict[str, Any]]:
    """Generate insights about duplicate rows."""

    duplicate_count = int(
        df.duplicated().sum()
    )

    if duplicate_count == 0:
        return []

    percentage = (
        duplicate_count / len(df) * 100
    )

    severity = (
        "high"
        if percentage >= 10
        else "medium"
        if percentage >= 2
        else "low"
    )

    return [
        {
            "category": "Data Quality",
            "title": "Duplicate records detected",
            "message": (
                f"{duplicate_count:,} duplicate rows "
                f"were found ({percentage:.1f}% of the dataset)."
            ),
            "severity": severity,
            "evidence": {
                "duplicate_count": duplicate_count,
                "duplicate_percentage": round(
                    percentage,
                    2,
                ),
            },
        }
    ]


def _numeric_distribution_insights(
    df: pd.DataFrame,
) -> list[dict[str, Any]]:
    """Identify strongly skewed numerical variables."""

    results = []

    numeric_columns = df.select_dtypes(
        include=np.number
    ).columns

    for column in numeric_columns:

        values = df[column].dropna()

        if len(values) < 3:
            continue

        skewness = float(values.skew())

        if abs(skewness) < 1:
            continue

        direction = (
            "right-skewed"
            if skewness > 0
            else "left-skewed"
        )

        results.append(
            {
                "category": "Distribution",
                "title": f"{column} is strongly skewed",
                "message": (
                    f"{column} is {direction} "
                    f"with skewness of {skewness:.2f}. "
                    f"The mean may not fully represent "
                    f"the typical observation."
                ),
                "severity": "medium",
                "evidence": {
                    "column": column,
                    "skewness": round(
                        skewness,
                        4,
                    ),
                },
            }
        )

    return results


def _correlation_insights(
    df: pd.DataFrame,
) -> list[dict[str, Any]]:
    """Identify strong numerical associations."""

    results = []

    numeric_df = df.select_dtypes(
        include=np.number
    )

    if numeric_df.shape[1] < 2:
        return results

    correlation = numeric_df.corr()

    columns = correlation.columns.tolist()

    for i in range(len(columns)):

        for j in range(i + 1, len(columns)):

            column_a = columns[i]
            column_b = columns[j]

            coefficient = correlation.loc[
                column_a,
                column_b,
            ]

            if pd.isna(coefficient):
                continue

            absolute_value = abs(
                float(coefficient)
            )

            if absolute_value < 0.7:
                continue

            direction = (
                "positive"
                if coefficient > 0
                else "negative"
            )

            results.append(
                {
                    "category": "Relationship",
                    "title": (
                        f"Strong {direction} "
                        f"association detected"
                    ),
                    "message": (
                        f"{column_a} and {column_b} "
                        f"have a correlation of "
                        f"{coefficient:.2f}. "
                        f"This indicates a strong "
                        f"{direction} association, "
                        f"not necessarily causation."
                    ),
                    "severity": "medium",
                    "evidence": {
                        "variable_1": column_a,
                        "variable_2": column_b,
                        "correlation": round(
                            float(coefficient),
                            4,
                        ),
                    },
                }
            )

    return results


def _categorical_insights(
    df: pd.DataFrame,
) -> list[dict[str, Any]]:
    """Identify dominant categories."""

    results = []

    categorical_columns = df.select_dtypes(
        include=[
            "object",
            "category",
            "bool",
        ]
    ).columns

    for column in categorical_columns:

        values = df[column].dropna()

        if values.empty:
            continue

        frequencies = (
            values.value_counts(
                normalize=True
            )
        )

        if frequencies.empty:
            continue

        dominant_value = frequencies.index[0]
        dominant_percentage = (
            frequencies.iloc[0] * 100
        )

        if dominant_percentage < 70:
            continue

        results.append(
            {
                "category": "Categorical Analysis",
                "title": (
                    f"{column} has a dominant category"
                ),
                "message": (
                    f"'{dominant_value}' represents "
                    f"{dominant_percentage:.1f}% "
                    f"of non-missing values in {column}."
                ),
                "severity": "low",
                "evidence": {
                    "column": column,
                    "dominant_value": str(
                        dominant_value
                    ),
                    "dominant_percentage": round(
                        dominant_percentage,
                        2,
                    ),
                },
            }
        )

    return results


def _potential_id_insights(
    df: pd.DataFrame,
) -> list[dict[str, Any]]:
    """Identify columns that may simply be identifiers."""

    results = []

    row_count = len(df)

    if row_count == 0:
        return results

    for column in df.columns:

        unique_ratio = (
            df[column].nunique(
                dropna=True
            ) / row_count
        )

        normalized = (
            str(column)
            .strip()
            .lower()
            .replace("-", "_")
            .replace(" ", "_")
        )

        looks_like_id = (
            normalized.endswith("_id")
            or normalized.endswith("id")
            or normalized in {
                "identifier",
                "customer_id",
                "customerid",
                "user_id",
                "userid",
                "transaction_id",
                "transactionid",
                "order_id",
                "orderid",
            }
        )

        if looks_like_id and unique_ratio >= 0.8:

            results.append(
                {
                    "category": "Modeling",
                    "title": (
                        f"{column} may be an identifier"
                    ),
                    "message": (
                        f"{column} has a high "
                        f"uniqueness ratio of "
                        f"{unique_ratio * 100:.1f}%. "
                        f"It may be an identifier rather "
                        f"than a useful predictive feature."
                    ),
                    "severity": "medium",
                    "evidence": {
                        "column": column,
                        "unique_ratio": round(
                            unique_ratio * 100,
                            2,
                        ),
                    },
                }
            )

    return results


def _sort_insights(
    insights: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Sort insights by importance."""

    severity_rank = {
        "critical": 0,
        "high": 1,
        "medium": 2,
        "low": 3,
    }

    return sorted(
        insights,
        key=lambda item: severity_rank.get(
            item.get("severity", "low"),
            3,
        ),
    )


def summarize_insights(
    insights: list[dict[str, Any]],
) -> dict[str, int]:
    """
    Return a compact count of insights by severity.
    """

    summary = {
        "critical": 0,
        "high": 0,
        "medium": 0,
        "low": 0,
        "total": len(insights),
    }

    for insight in insights:

        severity = insight.get(
            "severity",
            "low",
        )

        if severity in summary:
            summary[severity] += 1

    return summary
