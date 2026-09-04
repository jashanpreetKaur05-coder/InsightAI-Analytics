"""
InsightAI Analytics
===================

Interactive AI-ready business analytics dashboard.

Current capabilities:
- Multi-format dataset upload
- Dataset overview
- Data quality analysis
- Automated business insights
- Interactive visualizations
- Statistical analysis
- Data filtering
"""

import io

import pandas as pd
import streamlit as st

from modules.data_quality import (
    analyze_data_quality,
)

from modules.insights import (
    generate_insights,
)

from modules.statistics import (
    numeric_summary,
    strongest_correlations,
    detect_numeric_skewness,
)

from modules.visualizations import (
    create_histogram,
    create_boxplot,
    create_bar_chart,
    create_scatter_plot,
    create_correlation_heatmap,
    create_time_series,
)


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="InsightAI Analytics",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown(
    """
    <style>

    .main-title {
        font-size: 42px;
        font-weight: 700;
        margin-bottom: 5px;
    }

    .subtitle {
        font-size: 18px;
        color: #666666;
        margin-bottom: 25px;
    }

    .insight-card {
        padding: 16px;
        border-radius: 10px;
        border: 1px solid #dddddd;
        margin-bottom: 10px;
    }

    .section-title {
        font-size: 26px;
        font-weight: 600;
        margin-top: 20px;
    }

    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# SESSION STATE
# ============================================================

if "data" not in st.session_state:
    st.session_state.data = None

if "file_name" not in st.session_state:
    st.session_state.file_name = None


# ============================================================
# DATA LOADING
# ============================================================

def load_uploaded_file(
    uploaded_file,
) -> pd.DataFrame:
    """
    Load a supported uploaded dataset.

    Supported formats:
        CSV
        XLSX
        XLS
        JSON
        Parquet
    """

    if uploaded_file is None:
        raise ValueError("No file was uploaded.")

    file_name = uploaded_file.name.lower()

    file_bytes = uploaded_file.getvalue()

    if file_name.endswith(".csv"):

        try:
            return pd.read_csv(
                io.BytesIO(file_bytes)
            )
        except UnicodeDecodeError:

            return pd.read_csv(
                io.BytesIO(file_bytes),
                encoding="latin-1",
            )

    if file_name.endswith(".xlsx"):

        return pd.read_excel(
            io.BytesIO(file_bytes),
            engine="openpyxl",
        )

    if file_name.endswith(".xls"):

        return pd.read_excel(
            io.BytesIO(file_bytes)
        )

    if file_name.endswith(".json"):

        return pd.read_json(
            io.BytesIO(file_bytes)
        )

    if file_name.endswith(".parquet"):

        return pd.read_parquet(
            io.BytesIO(file_bytes)
        )

    raise ValueError(
        "Unsupported file format. "
        "Please upload CSV, Excel, JSON, or Parquet."
    )


# ============================================================
# HEADER
# ============================================================

st.markdown(
    '<div class="main-title">InsightAI Analytics</div>',
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="subtitle">
    Interactive business intelligence, automated data analysis,
    statistical insights and predictive analytics.
    </div>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.header("Dataset")

    uploaded_file = st.file_uploader(
        "Upload your dataset",
        type=[
            "csv",
            "xlsx",
            "xls",
            "json",
            "parquet",
        ],
        help=(
            "Upload a structured dataset for automated "
            "analysis."
        ),
    )

    if uploaded_file is not None:

        if (
            st.session_state.file_name
            != uploaded_file.name
        ):

            try:

                df = load_uploaded_file(
                    uploaded_file
                )

                st.session_state.data = df
                st.session_state.file_name = (
                    uploaded_file.name
                )

                st.success(
                    "Dataset loaded successfully."
                )

            except Exception as error:

                st.session_state.data = None
                st.session_state.file_name = None

                st.error(
                    f"Unable to load dataset: {error}"
                )

    st.divider()

    st.caption(
        "InsightAI Analytics"
    )

    st.caption(
        "Data stays inside the running application "
        "unless external AI/API services are explicitly "
        "connected later."
    )


# ============================================================
# CHECK DATASET
# ============================================================

df = st.session_state.data


if df is None:

    st.info(
        "Upload a dataset from the sidebar to begin analysis."
    )

    st.markdown(
        """
        ### What this application can analyze

        **Dataset structure**
        - Rows and columns
        - Data types
        - Numerical and categorical variables
        - Unique values

        **Data quality**
        - Missing values
        - Duplicate records
        - Constant columns
        - Potential identifiers
        - Outliers

        **Analytics**
        - Distributions
        - Correlations
        - Relationships
        - Category analysis
        - Statistical summaries

        **Coming next**
        - Natural-language data questions
        - Automatic target detection
        - Machine-learning predictions
        - Forecasting
        - Model comparison
        - Model evaluation
        """
    )

    st.stop()


# ============================================================
# BASIC VALIDATION
# ============================================================

if df.empty:

    st.error(
        "The uploaded dataset contains no rows."
    )

    st.stop()


# ============================================================
# DATASET HEADER
# ============================================================

st.success(
    f"Loaded: {st.session_state.file_name}"
)


# ============================================================
# KPI CARDS
# ============================================================

row_count = df.shape[0]
column_count = df.shape[1]

numeric_count = len(
    df.select_dtypes(
        include="number"
    ).columns
)

categorical_count = len(
    df.select_dtypes(
        include=[
            "object",
            "category",
            "bool",
        ]
    ).columns
)

missing_count = int(
    df.isna().sum().sum()
)

col1, col2, col3, col4, col5 = st.columns(5)

with col1:
    st.metric(
        "Rows",
        f"{row_count:,}",
    )

with col2:
    st.metric(
        "Columns",
        f"{column_count:,}",
    )

with col3:
    st.metric(
        "Numeric",
        f"{numeric_count:,}",
    )

with col4:
    st.metric(
        "Categorical",
        f"{categorical_count:,}",
    )

with col5:
    st.metric(
        "Missing Cells",
        f"{missing_count:,}",
    )


# ============================================================
# DATA QUALITY
# ============================================================

quality_results = analyze_data_quality(df)

quality_score = quality_results[
    "quality_score"
]

st.markdown(
    '<div class="section-title">Data Quality</div>',
    unsafe_allow_html=True,
)

quality_col1, quality_col2 = st.columns(
    [1, 2]
)

with quality_col1:

    st.metric(
        "Quality Score",
        f"{quality_score:.1f}/100",
    )

with quality_col2:

    if quality_score >= 90:

        st.success(
            "Dataset quality is relatively strong."
        )

    elif quality_score >= 70:

        st.warning(
            "Dataset has some quality issues "
            "that should be reviewed."
        )

    else:

        st.error(
            "Dataset contains significant quality issues."
        )


# ============================================================
# TABS
# ============================================================

tabs = st.tabs(
    [
        "Overview",
        "Data Quality",
        "Insights",
        "Explore",
        "Statistics",
        "Data",
    ]
)


# ============================================================
# OVERVIEW TAB
# ============================================================

with tabs[0]:

    st.subheader(
        "Dataset Overview"
    )

    overview_col1, overview_col2 = st.columns(
        2
    )

    with overview_col1:

        st.write(
            "### Column Types"
        )

        type_table = pd.DataFrame(
            {
                "Column": df.columns,
                "Data Type": [
                    str(dtype)
                    for dtype in df.dtypes
                ],
                "Non-Null": [
                    int(df[column].notna().sum())
                    for column in df.columns
                ],
                "Unique": [
                    int(
                        df[column].nunique(
                            dropna=True
                        )
                    )
                    for column in df.columns
                ],
            }
        )

        st.dataframe(
            type_table,
            use_container_width=True,
            hide_index=True,
        )

    with overview_col2:

        st.write(
            "### Numerical Summary"
        )

        summary = numeric_summary(df)

        if summary.empty:

            st.info(
                "No numerical columns detected."
            )

        else:

            st.dataframe(
                summary,
                use_container_width=True,
                hide_index=True,
            )


# ============================================================
# DATA QUALITY TAB
# ============================================================

with tabs[1]:

    st.subheader(
        "Data Quality Analysis"
    )

    missing_df = quality_results[
        "missing"
    ]

    st.write(
        "### Missing Values"
    )

    st.dataframe(
        missing_df,
        use_container_width=True,
        hide_index=True,
    )

    st.write(
        "### Duplicate Records"
    )

    duplicate_info = quality_results[
        "duplicates"
    ]

    duplicate_col1, duplicate_col2 = st.columns(
        2
    )

    with duplicate_col1:

        st.metric(
            "Duplicate Rows",
            duplicate_info[
                "duplicate_count"
            ],
        )

    with duplicate_col2:

        st.metric(
            "Duplicate %",
            f'{duplicate_info["duplicate_percentage"]:.2f}%',
        )

    st.write(
        "### Constant Columns"
    )

    constant_columns = quality_results[
        "constant_columns"
    ]

    if constant_columns:

        st.warning(
            f"Detected {len(constant_columns)} "
            "constant column(s)."
        )

        st.write(
            constant_columns
        )

    else:

        st.success(
            "No constant columns detected."
        )

    st.write(
        "### Potential Identifier Columns"
    )

    potential_ids = quality_results[
        "potential_id_columns"
    ]

    if potential_ids:

        st.info(
            "Potential identifier columns: "
            + ", ".join(
                map(str, potential_ids)
            )
        )

    else:

        st.write(
            "No obvious identifier columns detected."
        )

    st.write(
        "### Numeric Outliers"
    )

    outlier_df = quality_results[
        "outliers"
    ]

    if outlier_df.empty:

        st.info(
            "No numerical columns available "
            "for outlier analysis."
        )

    else:

        st.dataframe(
            outlier_df,
            use_container_width=True,
            hide_index=True,
        )


# ============================================================
# INSIGHTS TAB
# ============================================================

with tabs[2]:

    st.subheader(
        "Automated Business Insights"
    )

    insights = generate_insights(
        df,
        max_insights=20,
    )

    if not insights:

        st.success(
            "No major rule-based insights were detected."
        )

    else:

        for insight in insights:

            severity = insight[
                "severity"
            ]

            title = insight[
                "title"
            ]

            message = insight[
                "message"
            ]

            if severity == "critical":

                st.error(
                    f"🔴 {title}\n\n{message}"
                )

            elif severity == "high":

                st.warning(
                    f"🟠 {title}\n\n{message}"
                )

            elif severity == "medium":

                st.info(
                    f"🔵 {title}\n\n{message}"
                )

            else:

                st.write(
                    f"**{title}**\n\n{message}"
                )


# ============================================================
# EXPLORE TAB
# ============================================================

with tabs[3]:

    st.subheader(
        "Interactive Data Exploration"
    )

    numeric_columns = df.select_dtypes(
        include="number"
    ).columns.tolist()

    categorical_columns = df.select_dtypes(
        include=[
            "object",
            "category",
            "bool",
        ]
    ).columns.tolist()

    all_columns = df.columns.tolist()

    chart_type = st.selectbox(
        "Chart type",
        [
            "Histogram",
            "Box Plot",
            "Bar Chart",
            "Scatter Plot",
            "Correlation Heatmap",
            "Time Series",
        ],
    )

    if chart_type == "Histogram":

        if not numeric_columns:

            st.warning(
                "No numerical columns available."
            )

        else:

            column = st.selectbox(
                "Numerical column",
                numeric_columns,
            )

            bins = st.slider(
                "Number of bins",
                min_value=5,
                max_value=100,
                value=30,
            )

            fig = create_histogram(
                df,
                column,
                bins=bins,
            )

            st.plotly_chart(
                fig,
                use_container_width=True,
            )

    elif chart_type == "Box Plot":

        if not numeric_columns:

            st.warning(
                "No numerical columns available."
            )

        else:

            column = st.selectbox(
                "Numerical column",
                numeric_columns,
            )

            fig = create_boxplot(
                df,
                column,
            )

            st.plotly_chart(
                fig,
                use_container_width=True,
            )

    elif chart_type == "Bar Chart":

        if not categorical_columns:

            st.warning(
                "No categorical columns available."
            )

        else:

            category_column = st.selectbox(
                "Category column",
                categorical_columns,
            )

            use_value_column = st.checkbox(
                "Aggregate a numerical value",
                value=False,
            )

            value_column = None

            if use_value_column:

                if not numeric_columns:

                    st.warning(
                        "No numerical columns available."
                    )

                else:

                    value_column = st.selectbox(
                        "Value column",
                        numeric_columns,
                    )

            top_n = st.slider(
                "Top categories",
                min_value=5,
                max_value=50,
                value=15,
            )

            fig = create_bar_chart(
                df,
                category_column,
                value_column,
                top_n,
            )

            st.plotly_chart(
                fig,
                use_container_width=True,
            )

    elif chart_type == "Scatter Plot":

        if len(numeric_columns) < 2:

            st.warning(
                "At least two numerical columns "
                "are required."
            )

        else:

            x_column = st.selectbox(
                "X-axis",
                numeric_columns,
            )

            y_column = st.selectbox(
                "Y-axis",
                numeric_columns,
                index=(
                    1
                    if len(numeric_columns) > 1
                    else 0
                ),
            )

            color_column = st.selectbox(
                "Color by",
                ["None"] + categorical_columns,
            )

            if color_column == "None":
                color_column = None

            fig = create_scatter_plot(
                df,
                x_column,
                y_column,
                color_column,
            )

            st.plotly_chart(
                fig,
                use_container_width=True,
            )

    elif chart_type == "Correlation Heatmap":

        fig = create_correlation_heatmap(
            df
        )

        if fig is None:

            st.warning(
                "At least two numerical columns "
                "are required."
            )

        else:

            st.plotly_chart(
                fig,
                use_container_width=True,
            )

    elif chart_type == "Time Series":

        date_column = st.selectbox(
            "Date column",
            all_columns,
        )

        if not numeric_columns:

            st.warning(
                "No numerical value columns available."
            )

        else:

            value_column = st.selectbox(
                "Value column",
                numeric_columns,
            )

            aggregation = st.selectbox(
                "Aggregation",
                [
                    "sum",
                    "mean",
                    "median",
                    "min",
                    "max",
                    "count",
                ],
            )

            try:

                fig = create_time_series(
                    df,
                    date_column,
                    value_column,
                    aggregation,
                )

                st.plotly_chart(
                    fig,
                    use_container_width=True,
                )

            except Exception as error:

                st.error(
                    f"Unable to create time series: {error}"
                )


# ============================================================
# STATISTICS TAB
# ============================================================

with tabs[4]:

    st.subheader(
        "Statistical Analysis"
    )

    st.write(
        "### Strongest Numerical Relationships"
    )

    correlations = strongest_correlations(
        df,
        top_n=20,
    )

    if correlations.empty:

        st.info(
            "At least two numerical columns are "
            "required for correlation analysis."
        )

    else:

        st.dataframe(
            correlations,
            use_container_width=True,
            hide_index=True,
        )

    st.write(
        "### Numerical Skewness"
    )

    skewness = detect_numeric_skewness(
        df
    )

    if skewness.empty:

        st.info(
            "No numerical variables available."
        )

    else:

        st.dataframe(
            skewness,
            use_container_width=True,
            hide_index=True,
        )


# ============================================================
# DATA TAB
# ============================================================

with tabs[5]:

    st.subheader(
        "Dataset Explorer"
    )

    st.write(
        "Use the controls below to filter the dataset."
    )

    filtered_df = df.copy()

    filter_column = st.selectbox(
        "Filter column",
        ["None"] + all_columns,
    )

    if filter_column != "None":

        column = filtered_df[
            filter_column
        ]

        if pd.api.types.is_numeric_dtype(
            column
        ):

            minimum = float(
                column.min()
            )

            maximum = float(
                column.max()
            )

            if minimum != maximum:

                selected_range = st.slider(
                    "Value range",
                    min_value=minimum,
                    max_value=maximum,
                    value=(
                        minimum,
                        maximum,
                    ),
                )

                filtered_df = filtered_df[
                    filtered_df[
                        filter_column
                    ].between(
                        selected_range[0],
                        selected_range[1],
                    )
                ]

        else:

            values = (
                column
                .dropna()
                .astype(str)
                .unique()
                .tolist()
            )

            if len(values) <= 100:

                selected_values = st.multiselect(
                    "Select values",
                    options=sorted(values),
                )

                if selected_values:

                    filtered_df = filtered_df[
                        filtered_df[
                            filter_column
                        ].astype(str).isin(
                            selected_values
                        )
                    ]

    st.write(
        f"Showing {len(filtered_df):,} "
        f"of {len(df):,} rows"
    )

    st.dataframe(
        filtered_df,
        use_container_width=True,
        height=500,
    )


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "InsightAI Analytics | Automated analytics "
    "with transparent statistical methods."
)
