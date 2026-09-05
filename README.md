 ## 📊 InsightAI Analytics

## Interactive Business Intelligence & Data Analytics Platform

InsightAI Analytics is an interactive Python-based web application designed to transform structured datasets into meaningful business insights through automated data analysis, interactive visualizations, data-quality diagnostics, and statistical exploration.

The goal is to provide a single interface where users can upload a dataset, explore its structure, identify patterns and relationships, detect data-quality issues, and understand business performance without manually building every analysis from scratch.

---

## 🚀 Features

### 📌 Automated Dataset Profiling
Automatically analyzes an uploaded dataset and provides:

- Number of rows and columns
- Numerical and categorical variables
- Data types
- Unique values
- Missing values
- Duplicate records
- High-cardinality columns

### 📊 Interactive KPI Dashboard

Automatically calculates available business metrics such as:

- Revenue
- Profit
- Units Sold
- Profit Margin
- Average Customer Rating

KPIs dynamically respond to the selected filters.

### 📈 Trend Analysis

For datasets containing a valid date column, users can explore:

- Daily trends
- Weekly trends
- Monthly trends
- Revenue trends
- Profit trends
- Units sold trends
- Marketing spend trends
- Rolling averages

Interactive Plotly charts allow users to hover, zoom and explore specific periods.

### 🛍️ Product Analysis

Analyze product-level performance using:

- Revenue
- Profit
- Units Sold
- Returns
- Customer Ratings

Additional visualizations include:

- Product performance charts
- Product portfolio bubble charts
- Region → Product revenue hierarchy

### 🌍 Regional Analysis

Compare business performance across regions using:

- Revenue
- Profit
- Units Sold
- Returns
- Profit Margin

The application also provides regional KPI comparisons.

### 📢 Marketing Analytics

Analyze marketing channels using:

- Marketing Spend
- Revenue
- Profit
- Units Sold

The application calculates:

**ROAS = Revenue / Marketing Spend**

This allows users to compare marketing efficiency across channels.

### 👥 Customer Analysis

Analyze different customer segments based on:

- Revenue
- Profit
- Units Sold
- Returns
- Customer Rating

Customer Type × Product revenue can also be explored through an interactive heatmap.

### 🔵 Relationship Analysis

Explore relationships between numerical variables using interactive scatter plots.

Examples:

- Revenue vs Profit
- Marketing Spend vs Revenue
- Discount vs Profit
- Units Sold vs Revenue
- Area vs Price

The application can also display an OLS trendline and calculate Pearson correlation.

> Correlation is treated as an association and not as proof of causation.

### 📦 Distribution & Outlier Analysis

Explore numerical variables using:

- Histograms
- Box plots
- Violin plots

The application calculates:

- Mean
- Median
- Standard deviation
- Skewness
- IQR-based outliers

### 🧹 Data Quality Analysis

The platform evaluates:

- Missing values
- Missing-value percentage
- Duplicate records
- Data quality score
- High-cardinality variables

This helps users identify potential data-preparation issues before further analysis or modeling.

### 🧠 Automated Business Insights

The application generates rule-based analytical observations from the uploaded data.

Examples include:

- Revenue and profit relationships
- Strong correlations
- Highly skewed variables
- Highest-performing products
- Highest-performing marketing channels
- Missing-data warnings
- Profit-margin observations

The recommendations are generated from transparent statistical rules rather than unsupported assumptions.

### ⬇️ Export

Users can download the currently filtered dataset as a CSV file.

---

# 🛠️ Technology Stack

| Technology | Purpose |
|---|---|
| Python | Core programming and analytics |
| Pandas | Data manipulation and analysis |
| NumPy | Numerical computing |
| Plotly | Interactive visualizations |
| Streamlit | Web application interface |
| GitHub | Version control and project hosting |
| Streamlit Community Cloud | Application deployment |

---

# 📂 Suitable Datasets

InsightAI Analytics is designed for structured datasets containing a combination of numerical, categorical, and optionally date/time variables.

### 🛒 E-commerce / Retail

Examples:

- Order Date
- Product
- Category
- Region
- Customer Type
- Units Sold
- Revenue
- Profit
- Discount
- Marketing Spend
- Returns

This type of dataset can activate most of the dashboard's business-analysis capabilities.

### 📢 Marketing Analytics

Examples:

- Campaign
- Marketing Channel
- Advertising Spend
- Clicks
- Conversions
- Revenue
- Customer Segment

Useful for campaign performance and marketing efficiency analysis.

### 👥 HR Analytics

Examples:

- Department
- Job Role
- Salary
- Experience
- Performance
- Satisfaction
- Absences
- Attrition

Useful for workforce and employee analytics.

### 🏠 Real Estate

Examples:

- Property Price
- Area
- Location
- Bedrooms
- Bathrooms
- Year Built
- Condition

Useful for price relationships, distributions, location analysis and outlier detection.

### 📦 Product Analytics

Examples:

- Product
- Category
- Price
- Discount
- Rating
- Reviews
- Sales

Useful for product comparison and customer-review analysis.

---

# 🧩 How It Works

```text
             Upload Dataset
                    │
                    ▼
          Automated Data Profiling
                    │
          ┌─────────┴─────────┐
          ▼                   ▼
    Data Quality          Data Structure
          │                   │
          └─────────┬─────────┘
                    ▼
             Interactive Filters
                    │
                    ▼
             Business KPIs
                    │
       ┌────────────┼────────────┐
       ▼            ▼            ▼
     Trends      Products      Regions
       │            │            │
       ├────────────┼────────────┤
       ▼            ▼            ▼
   Marketing    Customers   Relationships
                    │
                    ▼
          Distributions & Outliers
                    │
                    ▼
          Automated Business Insights
                    │
                    ▼
             Export Filtered Data
