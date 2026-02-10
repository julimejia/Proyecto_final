import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import requests
import io
from datetime import datetime

st.set_page_config(page_title="Retail Sales Intelligent Dashboard", layout="wide")

# =====================================================
# CACHE
# =====================================================
@st.cache_data
def load_file(file):
    return pd.read_csv(file)

@st.cache_data
def load_url(url):
    response = requests.get(url)
    return pd.read_csv(io.StringIO(response.text))

# =====================================================
# CLEANING LOGIC (MEJORADA DEL NOTEBOOK)
# =====================================================
def clean_data(df, remove_duplicates, impute_method, outlier_threshold):

    df_original = df.copy()

    # eliminar duplicados
    if remove_duplicates:
        df = df.drop_duplicates()

    # convertir fecha
    if "Transaction Date" in df.columns:
        df["Transaction Date"] = pd.to_datetime(df["Transaction Date"], errors="coerce")

    # imputación
    numeric_cols = df.select_dtypes(include=np.number).columns

    if impute_method == "Media":
        df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].mean())
    elif impute_method == "Mediana":
        df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())
    elif impute_method == "Cero":
        df[numeric_cols] = df[numeric_cols].fillna(0)

    # outliers
    for col in numeric_cols:
        mean = df[col].mean()
        std = df[col].std()
        upper = mean + outlier_threshold * std
        lower = mean - outlier_threshold * std
        df = df[(df[col] <= upper) & (df[col] >= lower)]

    return df, df_original


# =====================================================
# SIDEBAR NAV
# =====================================================
st.sidebar.title("Navigation")
page = st.sidebar.radio(
    "Go to",
    ["ETL", "EDA", "KPIs", "AI Insights"]
)

st.sidebar.title("Data Source")

uploaded_file = st.sidebar.file_uploader("Upload CSV")
url_input = st.sidebar.text_input("Or URL")

# =====================================================
# LOAD DATA
# =====================================================
df = None

try:
    if uploaded_file:
        df = load_file(uploaded_file)
    elif url_input:
        df = load_url(url_input)
except:
    st.sidebar.error("Invalid file")

if df is None:
    st.stop()

# =====================================================
# ETL CONTROLS
# =====================================================
st.sidebar.title("ETL Controls")

remove_duplicates = st.sidebar.checkbox("Remove duplicates")

impute_method = st.sidebar.selectbox(
    "Imputation",
    ["Media", "Mediana", "Cero"]
)

outlier_threshold = st.sidebar.slider("Outlier threshold", 1.0, 5.0, 3.0)

df_clean, df_original = clean_data(df, remove_duplicates, impute_method, outlier_threshold)

# =====================================================
# GLOBAL FILTERS
# =====================================================
st.sidebar.title("Global Filters")

if "Category" in df_clean.columns:
    categories = st.sidebar.multiselect(
        "Category",
        df_clean["Category"].dropna().unique(),
        default=df_clean["Category"].dropna().unique()
    )
    df_clean = df_clean[df_clean["Category"].isin(categories)]

# =====================================================
# ETL PAGE
# =====================================================
if page == "ETL":

    st.title("ETL Interactive")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Before Cleaning")
        st.dataframe(df_original.head())

    with col2:
        st.subheader("After Cleaning")
        st.dataframe(df_clean.head())

    st.download_button("Download Clean CSV",
                       df_clean.to_csv(index=False),
                       "clean_data.csv")

# =====================================================
# EDA PAGE
# =====================================================
elif page == "EDA":

    st.title("Exploratory Data Analysis")

    numeric_cols = df_clean.select_dtypes(include=np.number).columns

    tab1, tab2, tab3 = st.tabs(["Univariate", "Bivariate", "Time Series"])

    with tab1:
        col = st.selectbox("Variable", numeric_cols)
        fig = px.histogram(df_clean, x=col)
        st.plotly_chart(fig, use_container_width=True)

    with tab2:
        col1 = st.selectbox("X", numeric_cols, key="x")
        col2 = st.selectbox("Y", numeric_cols, key="y")
        fig = px.scatter(df_clean, x=col1, y=col2)
        st.plotly_chart(fig, use_container_width=True)

    with tab3:
        if "Transaction Date" in df_clean.columns:
            ts = df_clean.groupby("Transaction Date")["Total Spent"].sum().reset_index()
            fig = px.line(ts, x="Transaction Date", y="Total Spent")
            st.plotly_chart(fig, use_container_width=True)

# =====================================================
# KPIS PAGE
# =====================================================
elif page == "KPIs":

    st.title("KPIs Dashboard")

    col1, col2, col3 = st.columns(3)

    col1.metric("Total Sales", f"${df_clean['Total Spent'].sum():,.0f}")
    col2.metric("Transactions", len(df_clean))
    col3.metric("Average Ticket", f"${df_clean['Total Spent'].mean():,.2f}")

# =====================================================
# AI INSIGHTS PAGE
# =====================================================
elif page == "AI Insights":

    st.title("AI Generated Insights")

    api_key = st.sidebar.text_input("Groq API Key", type="password")

    if st.button("Generate Insights"):

        if not api_key:
            st.warning("Insert API key")
        else:

            describe = df_clean.describe().to_string()

            prompt = f"""
Eres un analista de datos senior.

Datos:
{describe}

Proporciona:
1. 3 insights principales
2. 2 riesgos
3. 3 recomendaciones
4. 1 pregunta estratégica
"""

            headers = {
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json"
            }

            data = {
                "model": "llama3-70b-8192",
                "messages": [{"role": "user", "content": prompt}]
            }

            with st.spinner("Generating insights..."):
                r = requests.post("https://api.groq.com/openai/v1/chat/completions",
                                  headers=headers,
                                  json=data)

                if r.status_code == 200:
                    output = r.json()["choices"][0]["message"]["content"]
                    st.write(output)
                else:
                    st.error("LLM error")
# ================================
# GROQ REQUEST
# ================================
headers = {
    "Authorization": f"Bearer {api_key}",
    "Content-Type": "application/json"
}

payload = {
    "model": "llama3-70b-8192",
    "messages": [
        {"role": "user", "content": prompt}
    ],
    "temperature": 0.4,
    "max_tokens": 800
}

try:
    with st.spinner("Generating AI Insights..."):
        response = requests.post(
            "https://api.groq.com/openai/v1/chat/completions",
            headers=headers,
            json=payload,
            timeout=60
        )

    if response.status_code == 200:
        result = response.json()
        insights = result["choices"][0]["message"]["content"]

        st.subheader("AI Strategic Insights")
        st.write(insights)

        st.download_button(
            "Download Insights",
            insights,
            file_name="ai_insights.txt"
        )

    else:
        st.error(f"Groq API Error: {response.text}")

except Exception as e:
    st.error(f"Request failed: {e}")