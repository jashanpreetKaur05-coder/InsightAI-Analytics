"""
Data loading utilities for InsightAI Analytics.

This module handles supported structured data formats
and performs basic validation before analysis.
"""

from pathlib import Path

import pandas as pd


SUPPORTED_EXTENSIONS = {
    ".csv",
    ".xlsx",
    ".xls",
    ".json",
    ".parquet",
}


def load_dataset(uploaded_file) -> pd.DataFrame:
    """
    Load a dataset uploaded through Streamlit.

    Parameters
    ----------
    uploaded_file:
        Streamlit UploadedFile object.

    Returns
    -------
    pandas.DataFrame
        Loaded dataset.

    Raises
    ------
    ValueError
        If the file format is unsupported or the dataset is empty.
    """

    if uploaded_file is None:
        raise ValueError("No dataset was provided.")

    filename = uploaded_file.name
    extension = Path(filename).suffix.lower()

    if extension not in SUPPORTED_EXTENSIONS:
        supported = ", ".join(sorted(SUPPORTED_EXTENSIONS))

        raise ValueError(
            f"Unsupported file format: {extension}. "
            f"Supported formats: {supported}"
        )

    try:

        if extension == ".csv":
            df = pd.read_csv(uploaded_file)

        elif extension in {".xlsx", ".xls"}:
            df = pd.read_excel(uploaded_file)

        elif extension == ".json":
            df = pd.read_json(uploaded_file)

        elif extension == ".parquet":
            df = pd.read_parquet(uploaded_file)

        else:
            raise ValueError(
                f"Unsupported file format: {extension}"
            )

    except Exception as exc:

        raise ValueError(
            f"Could not read '{filename}'. "
            f"Please check that the file is valid. "
            f"Original error: {exc}"
        ) from exc

    if df.empty:

        raise ValueError(
            "The uploaded dataset contains no rows."
        )

    return df


def clean_column_names(df: pd.DataFrame) -> pd.DataFrame:
    """
    Standardize column names for safer analysis.

    Example:
        'Customer Name' -> 'customer_name'
        'Total Revenue' -> 'total_revenue'
    """

    cleaned_df = df.copy()

    cleaned_df.columns = (
        cleaned_df.columns
        .astype(str)
        .str.strip()
        .str.lower()
        .str.replace(r"[^a-z0-9]+", "_", regex=True)
        .str.strip("_")
    )

    return cleaned_df


def get_dataset_info(df: pd.DataFrame) -> dict:
    """
    Return basic information about the dataset.
    """

    return {
        "rows": int(df.shape[0]),
        "columns": int(df.shape[1]),
        "memory_mb": round(
            df.memory_usage(deep=True).sum() / (1024 ** 2),
            2,
        ),
        "column_names": df.columns.tolist(),
    }
