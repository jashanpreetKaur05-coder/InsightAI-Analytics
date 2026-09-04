"""
Visualization engine for InsightAI Analytics.

Creates reusable Plotly figures from a pandas DataFrame.
The functions are intentionally data-driven so the dashboard
can work with different business datasets.
"""

from typing import Optional

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go


def get_numeric_columns(df: pd.DataFrame) -> list[str]:
    """Return numerical columns."""
    return df.select_dtypes(include=np.number).columns.tolist()


def get_categorical_columns(df: pd.DataFrame) -> list[str]:
    """Return categorical/text columns."""
    return df.select_dtypes(
        include=["object", "category", "bool"]
    ).columns.tolist()


def get_datetime_columns(df: pd.DataFrame) -> list[str]:
    """Return datetime columns."""
    return df.select_dtypes(
        include=["datetime", "datetimetz"]
    ).columns.tolist()


def create_histogram(
    df: pd.DataFrame,
    column: str,
    bins: int = 30,
) -> go.Figure:
    """
    Create a distribution histogram for a numerical column.
    """

    if column not in df.columns:
        raise ValueError(f"Column '{column}' does not exist.")

    return px.histogram(
        df,
        x=column,
        nbins=bins,
        marginal="box",
        title=f"Distribution of {column}",
    )


def create_boxplot(
    df: pd.DataFrame,
    column: str,
) -> go.Figure:
    """
    Create a boxplot for a numerical column.
    """

    if column not in df.columns:
        raise ValueError(f"Column '{column}' does not exist.")

    return px.box(
        df,
        y=column,
        points="outliers",
        title=f"Box Plot of {column}",
    )


def create_bar_chart(
    df: pd.DataFrame,
    category_column: str,
    value_column: Optional[str] = None,
    top_n: int = 15,
) -> go.Figure:
    """
    Create a categorical bar chart.

    If value_column is supplied, values are aggregated using sum.
    Otherwise, category frequencies are displayed.
    """

    if category_column not in df.columns:
        raise ValueError(
            f"Column '{category_column}' does not exist."
        )

    if value_column is None:

        counts = (
            df[category_column]
            .value_counts(dropna=False)
            .head(top_n)
            .reset_index()
        )

        counts.columns = [
            category_column,
            "count",
        ]

        return px.bar(
            counts,
            x=category_column,
            y="count",
            title=f"Top {top_n} {category_column} Values",
        )

    if value_column not in df.columns:
        raise ValueError(
            f"Column '{value_column}' does not exist."
        )

    aggregated = (
        df.groupby(
            category_column,
            dropna=False,
        )[value_column]
        .sum()
        .sort_values(ascending=False)
        .head(top_n)
        .reset_index()
    )

    return px.bar(
        aggregated,
        x=category_column,
        y=value_column,
        title=(
            f"{value_column} by {category_column}"
        ),
    )


def create_scatter_plot(
    df: pd.DataFrame,
    x_column: str,
    y_column: str,
    color_column: Optional[str] = None,
) -> go.Figure:
    """
    Create a scatter plot to examine relationships
    between two numerical variables.
    """

    required_columns = {
        x_column,
        y_column,
    }

    if color_column:
        required_columns.add(color_column)

    missing = required_columns - set(df.columns)

    if missing:
        raise ValueError(
            f"Columns not found: {sorted(missing)}"
        )

    return px.scatter(
        df,
        x=x_column,
        y=y_column,
        color=color_column,
        title=f"{y_column} vs {x_column}",
        trendline="ols",
    )


def create_correlation_heatmap(
    df: pd.DataFrame,
) -> Optional[go.Figure]:
    """
    Create a correlation heatmap for numerical variables.

    Returns None when fewer than two numerical columns exist.
    """

    numeric_df = df.select_dtypes(
        include=np.number
    )

    if numeric_df.shape[1] < 2:
        return None

    correlation = numeric_df.corr(
        method="pearson"
    )

    return px.imshow(
        correlation,
        text_auto=".2f",
        aspect="auto",
        title="Numerical Correlation Matrix",
    )


def create_time_series(
    df: pd.DataFrame,
    date_column: str,
    value_column: str,
    aggregation: str = "sum",
) -> go.Figure:
    """
    Create a time-series chart.

    Supported aggregations:
        sum
        mean
        median
        min
        max
        count
    """

    if date_column not in df.columns:
        raise ValueError(
            f"Column '{date_column}' does not exist."
        )

    if value_column not in df.columns:
        raise ValueError(
            f"Column '{value_column}' does not exist."
        )

    working_df = df[
        [date_column, value_column]
    ].copy()

    working_df[date_column] = pd.to_datetime(
        working_df[date_column],
        errors="coerce",
    )

    working_df = working_df.dropna(
        subset=[date_column]
    )

    if working_df.empty:
        raise ValueError(
            "No valid dates were found."
        )

    aggregation_functions = {
        "sum": "sum",
        "mean": "mean",
        "median": "median",
        "min": "min",
        "max": "max",
        "count": "count",
    }

    if aggregation not in aggregation_functions:
        raise ValueError(
            "Invalid aggregation. Choose from: "
            + ", ".join(aggregation_functions)
        )

    function = aggregation_functions[aggregation]

    grouped = (
        working_df
        .set_index(date_column)[value_column]
        .resample("D")
        .agg(function)
        .reset_index()
    )

    return px.line(
        grouped,
        x=date_column,
        y=value_column,
        markers=True,
        title=(
            f"{aggregation.title()} of "
            f"{value_column} Over Time"
        ),
    )


def create_missing_values_chart(
    df: pd.DataFrame,
) -> go.Figure:
    """
    Visualize missing values by column.
    """

    missing = (
        df.isna()
        .sum()
        .sort_values(ascending=False)
    )

    missing = missing[missing > 0]

    if missing.empty:
        fig = go.Figure()

        fig.add_annotation(
            text="No missing values detected",
            x=0.5,
            y=0.5,
            xref="paper",
            yref="paper",
            showarrow=False,
        )

        fig.update_layout(
            title="Missing Values"
        )

        return fig

    result = missing.reset_index()

    result.columns = [
        "column",
        "missing_count",
    ]

    return px.bar(
        result,
        x="column",
        y="missing_count",
        title="Missing Values by Column",
    )


def create_auto_visualizations(
    df: pd.DataFrame,
) -> dict[str, go.Figure]:
    """
    Generate a small collection of useful visualizations
    automatically based on the dataset structure.

    This function does not attempt to visualize everything.
    It creates a sensible starting dashboard.
    """

    figures: dict[str, go.Figure] = {}

    numeric_columns = get_numeric_columns(df)
    categorical_columns = get_categorical_columns(df)

    # Distribution of the first numerical column.
    if numeric_columns:

        first_numeric = numeric_columns[0]

        figures["distribution"] = create_histogram(
            df,
            first_numeric,
        )

        figures["boxplot"] = create_boxplot(
            df,
            first_numeric,
        )

    # Top categories.
    if categorical_columns:

        first_category = categorical_columns[0]

        figures["category_distribution"] = create_bar_chart(
            df,
            first_category,
        )

    # Correlation analysis.
    correlation = create_correlation_heatmap(df)

    if correlation is not None:
        figures["correlation"] = correlation

    # Relationship between the first two numeric columns.
    if len(numeric_columns) >= 2:

        figures["relationship"] = create_scatter_plot(
            df,
            numeric_columns[0],
            numeric_columns[1],
        )

    # Missing values.
    figures["missing_values"] = (
        create_missing_values_chart(df)
    )

    return figures
