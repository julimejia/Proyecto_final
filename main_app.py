# =========================================================
# SMART RETAIL ANALYTICS DASHBOARD
# =========================================================

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

# =========================================================
# CONFIG
# =========================================================
st.set_page_config(layout="wide", page_title="Smart Retail Dashboard")

st.title("🧠 Smart Retail Analytics Dashboard")

# =========================================================
# SIDEBAR NAVIGATION
# =========================================================
st.sidebar.title("Navigation")

page = st.sidebar.radio(
    "Go to",
    ["ETL", "EDA", "KPIs", "AI Insights"]
)

# =========================================================
# FILE INPUT
# =========================================================
st.sidebar.subheader("Data Source")

uploaded_file = st.sidebar.file_uploader("Upload CSV", type=["csv"])

url_data = st.sidebar.text_input("Or paste dataset URL")

@st.cache_data
def load_data(file, url):
    if file:
        return pd.read_csv(file)
    if url:
        return pd.read_csv(url)
    return None

df_original = load_data(uploaded_file, url_data)

if df_original is None:
    st.warning("Upload dataset to continue")
    st.stop()

if "df" not in st.session_state:
    st.session_state.df = df_original.copy()

df = st.session_state.df

# =========================================================
# GLOBAL FILTERS
# =========================================================
st.sidebar.subheader("Global Filters")

if "Transaction Date" in df.columns:
    df["Transaction Date"] = pd.to_datetime(df["Transaction Date"], errors="coerce")

    min_date, max_date = df["Transaction Date"].min(), df["Transaction Date"].max()
    date_range = st.sidebar.date_input("Date range", [min_date, max_date])

    df = df[(df["Transaction Date"] >= pd.to_datetime(date_range[0])) &
            (df["Transaction Date"] <= pd.to_datetime(date_range[1]))]

if "Category" in df.columns:
    categories = st.sidebar.multiselect("Categories", df["Category"].unique(), default=df["Category"].unique())
    df = df[df["Category"].isin(categories)]

# =========================================================
# ======================== ETL TAB =========================
# =========================================================
if page == "ETL":

    st.header("Interactive ETL")

    col1, col2 = st.columns(2)

    with col1:
        remove_duplicates = st.checkbox("Remove duplicates")

        impute_method = st.selectbox(
            "Missing value method",
            ["None", "Mean", "Median", "Zero"]
        )

    with col2:
        outlier_threshold = st.slider("Outlier IQR threshold", 1.0, 3.0, 1.5)

    df_etl = df_original.copy()

    if remove_duplicates:
        df_etl = df_etl.drop_duplicates()

    numeric_cols = df_etl.select_dtypes(include=np.number).columns

    if impute_method != "None":
        for col in numeric_cols:
            if impute_method == "Mean":
                df_etl[col].fillna(df_etl[col].mean(), inplace=True)
            elif impute_method == "Median":
                df_etl[col].fillna(df_etl[col].median(), inplace=True)
            else:
                df_etl[col].fillna(0, inplace=True)

    # outlier filtering
    for col in numeric_cols:
        Q1 = df_etl[col].quantile(0.25)
        Q3 = df_etl[col].quantile(0.75)
        IQR = Q3 - Q1
        df_etl = df_etl[(df_etl[col] >= Q1 - outlier_threshold*IQR) &
                        (df_etl[col] <= Q3 + outlier_threshold*IQR)]

    st.session_state.df = df_etl

    st.write("Preview cleaned data")
    st.dataframe(df_etl.head())

# =========================================================
# ======================== EDA TAB =========================
# =========================================================
if page == "EDA":

    st.header("Exploratory Analysis")

    numeric_cols = df.select_dtypes(include=np.number).columns

    col = st.selectbox("Select variable", numeric_cols)

    fig = px.histogram(df, x=col)
    st.plotly_chart(fig, use_container_width=True)

    if len(numeric_cols) > 1:
        colx = st.selectbox("X", numeric_cols)
        coly = st.selectbox("Y", numeric_cols, index=1)

        fig = px.scatter(df, x=colx, y=coly, color="Category")
        st.plotly_chart(fig, use_container_width=True)

# =========================================================
# ======================== KPI TAB =========================
# =========================================================
if page == "KPIs":

    st.header("Business KPIs")

    revenue = df["Total Spent"].sum()
    avg_ticket = df["Total Spent"].mean()
    transactions = len(df)

    c1, c2, c3 = st.columns(3)
    c1.metric("Revenue", f"${revenue:,.0f}")
    c2.metric("Avg Ticket", f"${avg_ticket:,.2f}")
    c3.metric("Transactions", transactions)

    fig = px.bar(df.groupby("Category")["Total Spent"].sum().reset_index(),
                 x="Category", y="Total Spent")

    st.plotly_chart(fig, use_container_width=True)

# =========================================================
# ===================== AI INSIGHTS TAB ====================
# =========================================================
if page == "AI Insights":

    st.header("AI Generated Insights")

    api_key = st.sidebar.text_input("Groq API Key", type="password")

    if st.button("Generate Insights"):

        if not api_key:
            st.warning("Insert API key")
        else:
            from groq import Groq

            client = Groq(api_key=api_key)

            desc = df.describe().to_string()

            prompt = f"""
            Eres un analista senior.
            Analiza estos datos:

            {desc}

            Proporciona:
            1 insights
            2 riesgos
            3 recomendaciones
            """

            with st.spinner("Generating insights..."):
                completion = client.chat.completions.create(
                    model="llama-3.3-70b-versatile",
                    messages=[{"role":"user","content":prompt}]
                )

            st.write(completion.choices[0].message.content)

# =========================================================
# EXPORT
# =========================================================
st.sidebar.subheader("Export")

csv = df.to_csv(index=False).encode()
st.sidebar.download_button("Download CSV", csv, "clean_data.csv")
