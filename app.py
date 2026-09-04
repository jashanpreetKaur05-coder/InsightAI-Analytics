import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="InsightAI Analytics",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown("""
<style>
    .main {
        padding-top: 1rem;
    }

    .metric-card {
        background: #f8fafc;
        padding: 18px;
        border-radius: 12px;
        border: 1px solid #e5e7eb;
    }

    .insight-box {
        padding: 16px;
        border-radius: 10px;
        background: #eff6ff;
        border-left: 5px solid #2563eb;
        margin-bottom: 12px;
    }

    .recommendation-box {
        padding: 18px;
        border-radius: 10px;
        background: #f0fdf4;
        border-left: 5px solid #16a34a;
        margin-bottom: 12px;
    }

    h1 {
        font-weight: 700;
    }
</style>
""", unsafe_allow_html=True)


# ============================================================
# HEADER
# ============================================================

st.title("📊 InsightAI Analytics")
st.caption(
    "Interactive business intelligence, automated data analysis, "
    "statistical insights and predictive analytics."
)


# ============================================================
# DATA UPLOAD
# ============================================================

st.sidebar.header("📁 Dataset")

uploaded_file = st.sidebar.file_uploader(
    "Upload your CSV dataset",
    type=["csv"]
)

if uploaded_file is None:
    st.info("👆 Upload a CSV dataset from the sidebar to begin.")
    st.markdown("""
    ### What this dashboard provides

    - 📌 Business KPIs
    - 📈 Interactive trends
    - 🌍 Regional analysis
    - 🛍️ Product analysis
    - 📢 Marketing analysis
    - 👥 Customer analysis
    - 🔵 Correlation analysis
    - 📦 Outlier detection
    - 🧠 Automated business recommendations
    - 🧹 Data-quality assessment
    """)
    st.stop()


# ============================================================
# LOAD DATA
# ============================================================

try:
    df = pd.read_csv(uploaded_file)
except Exception as e:
    st.error(f"Unable to read the CSV file: {e}")
    st.stop()

df_original = df.copy()

st.sidebar.success("Dataset loaded successfully.")

# ============================================================
# BASIC CLEANING
# ============================================================

# Remove completely empty rows
df = df.dropna(how="all").copy()

# Convert Date if available
if "Date" in df.columns:
    try:
        df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    except Exception:
        pass


# ============================================================
# COLUMN IDENTIFICATION
# ============================================================

numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
categorical_cols = df.select_dtypes(
    include=["object", "category", "bool"]
).columns.tolist()

date_col = "Date" if "Date" in df.columns else None


# ============================================================
# SIDEBAR FILTERS
# ============================================================

st.sidebar.header("🎛️ Filters")

filtered_df = df.copy()

# Region
if "Region" in df.columns:
    regions = sorted(df["Region"].dropna().astype(str).unique())

    selected_regions = st.sidebar.multiselect(
        "Region",
        regions,
        default=regions
    )

    filtered_df = filtered_df[
        filtered_df["Region"].astype(str).isin(selected_regions)
    ]


# Product
if "Product" in df.columns:
    products = sorted(df["Product"].dropna().astype(str).unique())

    selected_products = st.sidebar.multiselect(
        "Product",
        products,
        default=products
    )

    filtered_df = filtered_df[
        filtered_df["Product"].astype(str).isin(selected_products)
    ]


# Customer Type
if "Customer_Type" in df.columns:
    customer_types = sorted(
        df["Customer_Type"].dropna().astype(str).unique()
    )

    selected_customer_types = st.sidebar.multiselect(
        "Customer Type",
        customer_types,
        default=customer_types
    )

    filtered_df = filtered_df[
        filtered_df["Customer_Type"].astype(str).isin(
            selected_customer_types
        )
    ]


# Marketing Channel
if "Marketing_Channel" in df.columns:
    channels = sorted(
        df["Marketing_Channel"].dropna().astype(str).unique()
    )

    selected_channels = st.sidebar.multiselect(
        "Marketing Channel",
        channels,
        default=channels
    )

    filtered_df = filtered_df[
        filtered_df["Marketing_Channel"].astype(str).isin(
            selected_channels
        )
    ]


# Date filter
if date_col and df[date_col].notna().any():

    min_date = df[date_col].min().date()
    max_date = df[date_col].max().date()

    date_range = st.sidebar.date_input(
        "Date Range",
        value=(min_date, max_date),
        min_value=min_date,
        max_value=max_date
    )

    if len(date_range) == 2:
        start_date, end_date = date_range

        filtered_df = filtered_df[
            (filtered_df[date_col].dt.date >= start_date)
            & (filtered_df[date_col].dt.date <= end_date)
        ]


# ============================================================
# MAIN KPI SECTION
# ============================================================

st.subheader("📌 Business Overview")

col1, col2, col3, col4, col5 = st.columns(5)


def safe_sum(column):
    if column in filtered_df.columns:
        return filtered_df[column].sum()
    return 0


def safe_mean(column):
    if column in filtered_df.columns:
        return filtered_df[column].mean()
    return 0


revenue = safe_sum("Revenue")
profit = safe_sum("Profit")
units = safe_sum("Units_Sold")
marketing = safe_sum("Marketing_Spend")
rating = safe_mean("Customer_Rating")

profit_margin = (
    (profit / revenue) * 100
    if revenue != 0
    else 0
)


with col1:
    st.metric(
        "💰 Revenue",
        f"{revenue:,.2f}"
    )

with col2:
    st.metric(
        "📈 Profit",
        f"{profit:,.2f}"
    )

with col3:
    st.metric(
        "📦 Units Sold",
        f"{units:,.0f}"
    )

with col4:
    st.metric(
        "💵 Profit Margin",
        f"{profit_margin:.1f}%"
    )

with col5:
    st.metric(
        "⭐ Avg Rating",
        f"{rating:.2f}"
    )


st.divider()


# ============================================================
# TABS
# ============================================================

tabs = st.tabs([
    "Overview",
    "📈 Trends",
    "🛍️ Products",
    "🌍 Regions",
    "📢 Marketing",
    "👥 Customers",
    "🔵 Relationships",
    "📦 Distributions",
    "🧹 Data Quality",
    "🧠 Insights"
])


# ============================================================
# OVERVIEW
# ============================================================

with tabs[0]:

    st.header("Dataset Overview")

    c1, c2, c3, c4, c5 = st.columns(5)

    c1.metric("Rows", f"{len(filtered_df):,}")
    c2.metric("Columns", f"{len(filtered_df.columns):,}")
    c3.metric("Numeric Columns", len(numeric_cols))
    c4.metric("Categorical Columns", len(categorical_cols))
    c5.metric(
        "Missing Cells",
        f"{filtered_df.isna().sum().sum():,}"
    )

    st.subheader("Column Types")

    column_info = pd.DataFrame({
        "Column": df.columns,
        "Data Type": df.dtypes.astype(str),
        "Non-Null": df.notna().sum().values,
        "Missing": df.isna().sum().values,
        "Unique": df.nunique(dropna=True).values
    })

    st.dataframe(
        column_info,
        use_container_width=True,
        hide_index=True
    )


# ============================================================
# TRENDS
# ============================================================

with tabs[1]:

    st.header("📈 Interactive Business Trends")

    if date_col and filtered_df[date_col].notna().any():

        metric_options = [
            col for col in
            ["Revenue", "Profit", "Units_Sold", "Marketing_Spend", "Returns"]
            if col in filtered_df.columns
        ]

        selected_metric = st.selectbox(
            "Select metric",
            metric_options
        )

        aggregation = st.selectbox(
            "Time aggregation",
            ["Daily", "Weekly", "Monthly"]
        )

        temp = filtered_df[
            [date_col, selected_metric]
        ].dropna()

        if aggregation == "Daily":
            temp["Period"] = temp[date_col].dt.date

        elif aggregation == "Weekly":
            temp["Period"] = (
                temp[date_col]
                .dt.to_period("W")
                .apply(lambda x: x.start_time)
            )

        else:
            temp["Period"] = (
                temp[date_col]
                .dt.to_period("M")
                .apply(lambda x: x.start_time)
            )

        trend = (
            temp.groupby("Period")[selected_metric]
            .sum()
            .reset_index()
        )

        fig = px.line(
            trend,
            x="Period",
            y=selected_metric,
            markers=True,
            title=f"{selected_metric} Over Time"
        )

        fig.update_layout(
            hovermode="x unified",
            height=500
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

    else:
        st.warning("A valid Date column is required for trend analysis.")


# ============================================================
# PRODUCTS
# ============================================================

with tabs[2]:

    st.header("🛍️ Product Performance")

    if "Product" in filtered_df.columns:

        product_metric = st.selectbox(
            "Metric",
            [
                col for col in
                ["Revenue", "Profit", "Units_Sold",
                 "Marketing_Spend", "Returns", "Customer_Rating"]
                if col in filtered_df.columns
            ],
            key="product_metric"
        )

        product_data = (
            filtered_df
            .groupby("Product")[product_metric]
            .agg(["sum", "mean"])
            .reset_index()
        )

        product_data = product_data.sort_values(
            "sum",
            ascending=False
        )

        fig = px.bar(
            product_data,
            x="Product",
            y="sum",
            title=f"{product_metric} by Product",
            text_auto=".2s"
        )

        fig.update_layout(
            xaxis_title="Product",
            yaxis_title=product_metric,
            height=500
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

        st.subheader("Product Summary")

        summary_cols = [
            c for c in [
                "Units_Sold",
                "Revenue",
                "Profit",
                "Returns",
                "Customer_Rating"
            ]
            if c in filtered_df.columns
        ]

        product_summary = (
            filtered_df
            .groupby("Product")[summary_cols]
            .agg({
                c: "mean" if c == "Customer_Rating"
                else "sum"
                for c in summary_cols
            })
            .reset_index()
        )

        st.dataframe(
            product_summary,
            use_container_width=True,
            hide_index=True
        )


# ============================================================
# REGIONS
# ============================================================

with tabs[3]:

    st.header("🌍 Regional Performance")

    if "Region" in filtered_df.columns:

        region_metric = st.selectbox(
            "Select metric",
            [
                c for c in
                ["Revenue", "Profit", "Units_Sold",
                 "Marketing_Spend", "Returns"]
                if c in filtered_df.columns
            ],
            key="region_metric"
        )

        region_data = (
            filtered_df
            .groupby("Region")[region_metric]
            .sum()
            .reset_index()
            .sort_values(region_metric, ascending=False)
        )

        fig = px.bar(
            region_data,
            x="Region",
            y=region_metric,
            text_auto=".2s",
            title=f"{region_metric} by Region"
        )

        fig.update_layout(height=500)

        st.plotly_chart(
            fig,
            use_container_width=True
        )

        # Revenue share
        if region_metric == "Revenue":

            st.subheader("Revenue Contribution")

            pie = px.pie(
                region_data,
                names="Region",
                values="Revenue",
                hole=0.45,
                title="Revenue Share by Region"
            )

            st.plotly_chart(
                pie,
                use_container_width=True
            )


# ============================================================
# MARKETING
# ============================================================

with tabs[4]:

    st.header("📢 Marketing Channel Analysis")

    if "Marketing_Channel" in filtered_df.columns:

        channel_metric = st.selectbox(
            "Metric",
            [
                c for c in
                ["Revenue", "Profit", "Units_Sold",
                 "Marketing_Spend"]
                if c in filtered_df.columns
            ],
            key="channel_metric"
        )

        channel_data = (
            filtered_df
            .groupby("Marketing_Channel")[channel_metric]
            .sum()
            .reset_index()
            .sort_values(
                channel_metric,
                ascending=False
            )
        )

        fig = px.bar(
            channel_data,
            x="Marketing_Channel",
            y=channel_metric,
            text_auto=".2s",
            title=f"{channel_metric} by Marketing Channel"
        )

        fig.update_layout(height=500)

        st.plotly_chart(
            fig,
            use_container_width=True
        )

        # Marketing efficiency
        if (
            "Revenue" in filtered_df.columns
            and "Marketing_Spend" in filtered_df.columns
        ):

            efficiency = (
                filtered_df
                .groupby("Marketing_Channel")
                .agg({
                    "Revenue": "sum",
                    "Marketing_Spend": "sum"
                })
                .reset_index()
            )

            efficiency["ROAS"] = (
                efficiency["Revenue"]
                / efficiency["Marketing_Spend"].replace(0, np.nan)
            )

            st.subheader("Marketing Efficiency")

            fig2 = px.bar(
                efficiency.sort_values(
                    "ROAS",
                    ascending=False
                ),
                x="Marketing_Channel",
                y="ROAS",
                text_auto=".2f",
                title="Revenue / Marketing Spend"
            )

            fig2.update_layout(height=450)

            st.plotly_chart(
                fig2,
                use_container_width=True
            )


# ============================================================
# CUSTOMERS
# ============================================================

with tabs[5]:

    st.header("👥 Customer Analysis")

    if "Customer_Type" in filtered_df.columns:

        customer_metric = st.selectbox(
            "Metric",
            [
                c for c in
                ["Revenue", "Profit", "Units_Sold",
                 "Returns", "Customer_Rating"]
                if c in filtered_df.columns
            ],
            key="customer_metric"
        )

        customer_data = (
            filtered_df
            .groupby("Customer_Type")[customer_metric]
            .agg(["sum", "mean"])
            .reset_index()
        )

        fig = px.bar(
            customer_data,
            x="Customer_Type",
            y="sum",
            text_auto=".2s",
            title=f"{customer_metric} by Customer Type"
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

        st.subheader("Customer Summary")

        st.dataframe(
            customer_data,
            use_container_width=True,
            hide_index=True
        )


# ============================================================
# RELATIONSHIPS
# ============================================================

with tabs[6]:

    st.header("🔵 Numerical Relationships")

    relationship_cols = [
        c for c in [
            "Revenue",
            "Profit",
            "Units_Sold",
            "Unit_Price",
            "Discount_Pct",
            "Marketing_Spend",
            "Returns",
            "Inventory_End",
            "Customer_Rating"
        ]
        if c in filtered_df.columns
    ]

    if len(relationship_cols) >= 2:

        c1, c2 = st.columns(2)

        with c1:
            x_variable = st.selectbox(
                "X-axis",
                relationship_cols,
                index=(
                    relationship_cols.index("Revenue")
                    if "Revenue" in relationship_cols
                    else 0
                )
            )

        with c2:
            y_variable = st.selectbox(
                "Y-axis",
                relationship_cols,
                index=(
                    relationship_cols.index("Profit")
                    if "Profit" in relationship_cols
                    else 1
                )
            )

        color_options = ["None"]

        for col in [
            "Region",
            "Product",
            "Customer_Type",
            "Marketing_Channel"
        ]:
            if col in filtered_df.columns:
                color_options.append(col)

        color_by = st.selectbox(
            "Color points by",
            color_options
        )

        plot_df = filtered_df[
            [x_variable, y_variable]
            + ([] if color_by == "None" else [color_by])
        ].dropna()

        fig = px.scatter(
            plot_df,
            x=x_variable,
            y=y_variable,
            color=None if color_by == "None" else color_by,
            hover_data=plot_df.columns,
            title=f"{y_variable} vs {x_variable}"
        )

        fig.update_traces(
            marker=dict(
                size=9,
                opacity=0.7
            )
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

        correlation = (
            plot_df[x_variable]
            .corr(plot_df[y_variable])
        )

        if correlation >= 0.7:
            strength = "Strong positive"
        elif correlation >= 0.4:
            strength = "Moderate positive"
        elif correlation >= 0.2:
            strength = "Weak positive"
        elif correlation <= -0.7:
            strength = "Strong negative"
        elif correlation <= -0.4:
            strength = "Moderate negative"
        elif correlation <= -0.2:
            strength = "Weak negative"
        else:
            strength = "Very weak"

        st.info(
            f"Correlation between **{x_variable}** and "
            f"**{y_variable}**: **{correlation:.3f}** "
            f"({strength}). Correlation does not imply causation."
        )

        # Correlation heatmap
        st.subheader("Correlation Heatmap")

        corr = filtered_df[relationship_cols].corr()

        fig_heatmap = px.imshow(
            corr,
            text_auto=".2f",
            aspect="auto",
            title="Numerical Correlation Matrix"
        )

        fig_heatmap.update_layout(
            height=650
        )

        st.plotly_chart(
            fig_heatmap,
            use_container_width=True
        )


# ============================================================
# DISTRIBUTIONS
# ============================================================

with tabs[7]:

    st.header("📦 Distribution & Outlier Explorer")

    if numeric_cols:

        selected_distribution = st.selectbox(
            "Select numerical column",
            numeric_cols
        )

        chart_type = st.selectbox(
            "Visualization",
            [
                "Histogram",
                "Box Plot"
            ]
        )

        values = filtered_df[
            selected_distribution
        ].dropna()

        if chart_type == "Histogram":

            bins = st.slider(
                "Number of bins",
                5,
                100,
                30
            )

            fig = px.histogram(
                values,
                x=selected_distribution,
                nbins=bins,
                marginal="box",
                title=f"Distribution of {selected_distribution}"
            )

        else:

            fig = px.box(
                values,
                y=selected_distribution,
                points="outliers",
                title=f"Box Plot of {selected_distribution}"
            )

        fig.update_layout(
            height=550
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

        # Statistics
        c1, c2, c3, c4 = st.columns(4)

        c1.metric(
            "Mean",
            f"{values.mean():,.2f}"
        )

        c2.metric(
            "Median",
            f"{values.median():,.2f}"
        )

        c3.metric(
            "Std Dev",
            f"{values.std():,.2f}"
        )

        c4.metric(
            "Skewness",
            f"{values.skew():,.2f}"
        )


# ============================================================
# DATA QUALITY
# ============================================================

with tabs[8]:

    st.header("🧹 Data Quality Analysis")

    total_cells = (
        len(filtered_df)
        * len(filtered_df.columns)
    )

    missing_cells = (
        filtered_df.isna().sum().sum()
    )

    quality_score = (
        100 - (missing_cells / total_cells * 100)
        if total_cells > 0
        else 0
    )

    c1, c2, c3 = st.columns(3)

    c1.metric(
        "Quality Score",
        f"{quality_score:.1f}/100"
    )

    c2.metric(
        "Missing Cells",
        f"{missing_cells:,}"
    )

    duplicate_count = filtered_df.duplicated().sum()

    c3.metric(
        "Duplicate Rows",
        f"{duplicate_count:,}"
    )

    st.subheader("Missing Values")

    missing = pd.DataFrame({
        "Column": filtered_df.columns,
        "Missing Count": filtered_df.isna().sum().values
    })

    missing["Missing %"] = (
        missing["Missing Count"]
        / len(filtered_df)
        * 100
    )

    missing["Severity"] = np.select(
        [
            missing["Missing %"] == 0,
            missing["Missing %"] < 5,
            missing["Missing %"] < 15
        ],
        [
            "None",
            "Low",
            "Medium"
        ],
        default="High"
    )

    missing = missing.sort_values(
        "Missing Count",
        ascending=False
    )

    st.dataframe(
        missing,
        use_container_width=True,
        hide_index=True
    )

    st.subheader("Duplicate Records")

    if duplicate_count == 0:
        st.success("No duplicate records detected.")
    else:
        st.warning(
            f"{duplicate_count} duplicate rows detected."
        )

    # High cardinality
    st.subheader("High-Cardinality Columns")

    cardinality = []

    for col in df.columns:

        unique_count = df[col].nunique(
            dropna=True
        )

        unique_ratio = (
            unique_count / len(df)
            if len(df) > 0
            else 0
        )

        if unique_ratio >= 0.8:
            cardinality.append({
                "Column": col,
                "Unique Values": unique_count,
                "Unique Ratio": unique_ratio
            })

    if cardinality:

        cardinality_df = pd.DataFrame(
            cardinality
        )

        cardinality_df["Unique Ratio"] *= 100

        st.dataframe(
            cardinality_df,
            use_container_width=True,
            hide_index=True
        )

        st.caption(
            "High cardinality means a large proportion of values "
            "are unique. This does not automatically mean the "
            "column is an identifier."
        )

    else:
        st.success(
            "No unusually high-cardinality columns detected."
        )


# ============================================================
# AUTOMATED INSIGHTS
# ============================================================

with tabs[9]:

    st.header("🧠 Automated Business Insights")

    insights = []
    recommendations = []

    # --------------------------------------------------------
    # Revenue / Profit
    # --------------------------------------------------------

    if "Revenue" in filtered_df.columns:

        total_revenue = filtered_df["Revenue"].sum()

        if total_revenue > 0:
            insights.append(
                f"Total revenue for the selected data is "
                f"**{total_revenue:,.2f}**."
            )

    if "Profit" in filtered_df.columns:

        total_profit = filtered_df["Profit"].sum()

        if total_profit > 0:

            insights.append(
                f"Total profit is **{total_profit:,.2f}**, "
                f"with an estimated profit margin of "
                f"**{profit_margin:.1f}%**."
            )

            if profit_margin < 10:

                recommendations.append(
                    "Profit margin is relatively low. "
                    "Review pricing, discounts, returns and "
                    "marketing costs to identify margin pressure."
                )

            elif profit_margin > 25:

                recommendations.append(
                    "Profit margin is strong. Investigate which "
                    "products, regions and channels are contributing "
                    "most to this performance."
                )


    # --------------------------------------------------------
    # Correlation
    # --------------------------------------------------------

    if (
        "Revenue" in filtered_df.columns
        and "Profit" in filtered_df.columns
    ):

        corr = filtered_df[
            ["Revenue", "Profit"]
        ].corr().iloc[0, 1]

        if abs(corr) >= 0.7:

            insights.append(
                f"Revenue and Profit show a strong correlation "
                f"of **{corr:.2f}**."
            )

            recommendations.append(
                "Because revenue and profit move closely together, "
                "investigate the factors that drive revenue while "
                "protecting the current profit margin."
            )


    # --------------------------------------------------------
    # Skewness
    # --------------------------------------------------------

    for col in numeric_cols:

        if col not in filtered_df.columns:
            continue

        series = filtered_df[col].dropna()

        if len(series) < 3:
            continue

        skew = series.skew()

        if abs(skew) > 1:

            direction = (
                "right-skewed"
                if skew > 0
                else "left-skewed"
            )

            insights.append(
                f"**{col}** is strongly {direction} "
                f"with skewness of **{skew:.2f}**."
            )


    # --------------------------------------------------------
    # Missing values
    # --------------------------------------------------------

    missing_cols = (
        filtered_df.isna().sum()
    )

    missing_cols = missing_cols[
        missing_cols > 0
    ]

    for col, count in missing_cols.items():

        percentage = (
            count / len(filtered_df) * 100
        )

        if percentage >= 5:

            recommendations.append(
                f"Review missing values in **{col}** "
                f"({percentage:.1f}% of rows) before using "
                f"this variable for modeling."
            )


    # --------------------------------------------------------
    # Marketing
    # --------------------------------------------------------

    if (
        "Marketing_Channel" in filtered_df.columns
        and "Revenue" in filtered_df.columns
    ):

        channel_revenue = (
            filtered_df
            .groupby("Marketing_Channel")["Revenue"]
            .sum()
            .sort_values(ascending=False)
        )

        if len(channel_revenue) > 0:

            best_channel = channel_revenue.index[0]

            insights.append(
                f"**{best_channel}** generates the highest "
                f"total revenue among the selected marketing channels."
            )

            recommendations.append(
                f"Evaluate the campaigns associated with "
                f"**{best_channel}** and identify which characteristics "
                f"could be replicated in lower-performing channels."
            )


    # --------------------------------------------------------
    # Product
    # --------------------------------------------------------

    if (
        "Product" in filtered_df.columns
        and "Revenue" in filtered_df.columns
    ):

        product_revenue = (
            filtered_df
            .groupby("Product")["Revenue"]
            .sum()
            .sort_values(ascending=False)
        )

        if len(product_revenue) > 0:

            best_product = product_revenue.index[0]

            insights.append(
                f"**{best_product}** is the highest-revenue "
                f"product in the selected dataset."
            )

            recommendations.append(
                f"Investigate pricing, demand and marketing "
                f"patterns behind **{best_product}**'s performance."
            )


    # --------------------------------------------------------
    # DISPLAY
    # --------------------------------------------------------

    if insights:

        st.subheader("Key Findings")

        for insight in insights:

            st.markdown(
                f"""
                <div class="insight-box">
                    🔎 {insight}
                </div>
                """,
                unsafe_allow_html=True
            )

    else:

        st.info(
            "Not enough information to generate automated insights."
        )


    if recommendations:

        st.subheader("💡 Business Recommendations")

        for recommendation in recommendations:

            st.markdown(
                f"""
                <div class="recommendation-box">
                    💡 {recommendation}
                </div>
                """,
                unsafe_allow_html=True
            )


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "InsightAI Analytics | Interactive analytics with transparent "
    "statistical methods."
)
