# =====================================================
# IMPORTS & CONFIG
# =====================================================
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import requests
import io
from datetime import datetime

st.set_page_config(page_title="Retail Sales Intelligent Dashboard", layout="wide")

# =====================================================
# CACHE FUNCTIONS
# =====================================================
@st.cache_data
def load_file(file):
    return pd.read_csv(file)

@st.cache_data
def load_url(url):
    response = requests.get(url)
    return pd.read_csv(io.StringIO(response.text))

# =====================================================
# CLEANING LOGIC (IMPROVED FROM NOTEBOOK)
# =====================================================
def clean_data(df, remove_duplicates, impute_method, outlier_threshold):
    df_original = df.copy()

    # Eliminar duplicados
    if remove_duplicates:
        df = df.drop_duplicates()

    # Convertir fecha
    if "Transaction Date" in df.columns:
        df["Transaction Date"] = pd.to_datetime(df["Transaction Date"], errors="coerce")

    # Imputación
    numeric_cols = df.select_dtypes(include=np.number).columns

    if impute_method == "Media":
        df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].mean())
    elif impute_method == "Mediana":
        df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())
    elif impute_method == "Cero":
        df[numeric_cols] = df[numeric_cols].fillna(0)

    # Manejo de outliers
    for col in numeric_cols:
        mean = df[col].mean()
        std = df[col].std()
        upper = mean + outlier_threshold * std
        lower = mean - outlier_threshold * std
        df = df[(df[col] <= upper) & (df[col] >= lower)]

    return df, df_original

# =====================================================
# SIDEBAR NAVIGATION
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
# DATA LOADING
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
# EDA PAGE (ENHANCED)
# =====================================================
elif page == "EDA":
    st.title("Exploratory Data Analysis")

    # Sección 1: Información general
    st.header("1. Información General")
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Primeras 5 filas")
        st.dataframe(df_clean.head())
    
    with col2:
        st.subheader("Tipos de datos y valores no nulos")
        buffer = io.StringIO()
        df_clean.info(buf=buffer)
        st.text(buffer.getvalue())
    
    st.subheader("Estadísticas descriptivas")
    st.dataframe(df_clean.describe())

    # Sección 2: Valores faltantes
    st.header("2. Valores Faltantes")
    missing_count = df_clean.isnull().sum()
    missing_percentage = (df_clean.isnull().sum() / len(df_clean)) * 100
    missing_data = pd.DataFrame({
        'Missing Count': missing_count,
        'Missing Percentage': missing_percentage
    }).sort_values(by='Missing Percentage', ascending=False)
    
    missing_data = missing_data[missing_data['Missing Count'] > 0]
    
    if not missing_data.empty:
        st.dataframe(missing_data)
    else:
        st.success("✅ No hay valores faltantes en el dataset")

    # Sección 3: Filas duplicadas
    st.header("3. Filas Duplicadas")
    duplicate_rows_count = df_clean.duplicated().sum()
    if duplicate_rows_count > 0:
        st.warning(f"Se encontraron {duplicate_rows_count} filas duplicadas")
        st.dataframe(df_clean[df_clean.duplicated()])
    else:
        st.success("✅ No hay filas duplicadas")

    # Sección 4: Visualizaciones
    st.header("4. Visualizaciones")
    
    numeric_cols = df_clean.select_dtypes(include=np.number).columns
    
    tab1, tab2, tab3 = st.tabs(["Univariate", "Bivariate", "Time Series"])
    
    with tab1:
        if len(numeric_cols) > 0:
            col = st.selectbox("Selecciona variable numérica", numeric_cols)
            fig = px.histogram(df_clean, x=col, title=f'Distribución de {col}')
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("No hay columnas numéricas para visualizar")
    
    with tab2:
        if len(numeric_cols) > 1:
            col1 = st.selectbox("Variable X", numeric_cols, key="x")
            col2 = st.selectbox("Variable Y", numeric_cols, key="y")
            fig = px.scatter(df_clean, x=col1, y=col2, 
                           title=f'{col1} vs {col2}')
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("Se necesitan al menos 2 columnas numéricas")
    
    with tab3:
        if "Transaction Date" in df_clean.columns:
            ts_data = df_clean.copy()
            ts_data = ts_data.sort_values("Transaction Date")
            
            # Agregar opciones de resample
            resample_option = st.selectbox(
                "Frecuencia de agregación",
                ["Diario", "Semanal", "Mensual", "Trimestral", "Anual"]
            )
            
            freq_map = {
                "Diario": "D",
                "Semanal": "W",
                "Mensual": "M",
                "Trimestral": "Q",
                "Anual": "Y"
            }
            
            ts_grouped = ts_data.set_index("Transaction Date")["Total Spent"]
            ts_resampled = ts_grouped.resample(freq_map[resample_option]).sum().reset_index()
            
            fig = px.line(ts_resampled, x="Transaction Date", y="Total Spent",
                         title=f'Ventas Totales ({resample_option.lower()})')
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("No hay columna 'Transaction Date' para análisis temporal")

    # Sección 5: Análisis de categorías
    st.header("5. Análisis por Categoría")
    if "Category" in df_clean.columns and "Total Spent" in df_clean.columns:
        category_sales = df_clean.groupby("Category")["Total Spent"].agg(['sum', 'mean', 'count']).reset_index()
        category_sales = category_sales.rename(columns={
            'sum': 'Ventas Totales',
            'mean': 'Ticket Promedio',
            'count': 'Transacciones'
        })
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Métricas por Categoría")
            st.dataframe(category_sales)
        
        with col2:
            fig = px.bar(category_sales.sort_values("Ventas Totales", ascending=False),
                        x="Category", y="Ventas Totales",
                        title="Ventas Totales por Categoría")
            st.plotly_chart(fig, use_container_width=True)

# =====================================================
# KPIs PAGE
# =====================================================
elif page == "KPIs":
    st.title("KPIs Dashboard")
    
    # KPIs principales
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        total_sales = df_clean['Total Spent'].sum()
        st.metric("Total Sales", f"${total_sales:,.0f}")
    
    with col2:
        total_transactions = len(df_clean)
        st.metric("Transactions", f"{total_transactions:,}")
    
    with col3:
        avg_ticket = df_clean['Total Spent'].mean()
        st.metric("Average Ticket", f"${avg_ticket:,.2f}")
    
    with col4:
        unique_customers = df_clean['Customer ID'].nunique() if 'Customer ID' in df_clean.columns else "N/A"
        st.metric("Unique Customers", f"{unique_customers}")
    
    # KPIs secundarios
    st.subheader("Detailed Metrics")
    
    col1, col2 = st.columns(2)
    
    with col1:
        if "Quantity" in df_clean.columns and "Total Spent" in df_clean.columns:
            avg_price = df_clean['Total Spent'].sum() / df_clean['Quantity'].sum()
            st.metric("Average Price per Unit", f"${avg_price:.2f}")
        
        if "Discount Applied" in df_clean.columns:
            discount_rate = df_clean['Discount Applied'].mean() * 100
            st.metric("Discount Rate", f"{discount_rate:.1f}%")
    
    with col2:
        if "Payment Method" in df_clean.columns:
            payment_dist = df_clean['Payment Method'].value_counts()
            st.write("Payment Method Distribution:")
            st.dataframe(payment_dist)
        
        if "Location" in df_clean.columns:
            location_dist = df_clean['Location'].value_counts()
            st.write("Sales by Location:")
            st.dataframe(location_dist)

# =====================================================
# AI INSIGHTS PAGE
# =====================================================
elif page == "AI Insights":
    st.title("AI Generated Insights")
    
    api_key = st.sidebar.text_input("Groq API Key", type="password")
    
    if st.button("Generate Insights"):
        if not api_key:
            st.warning("Please enter your Groq API Key")
        else:
            # Preparar datos descriptivos
            describe = df_clean.describe().to_string()
            missing_info = missing_data.to_string() if 'missing_data' in locals() else "No missing values"
            duplicate_info = f"Duplicate rows: {duplicate_rows_count}" if 'duplicate_rows_count' in locals() else ""
            
            prompt = f"""
Eres un analista de datos senior experto en retail. Analiza los siguientes datos y proporciona insights estratégicos:

DATOS DEL DATASET:
{describe}

INFORMACIÓN DE CALIDAD:
{missing_info}
{duplicate_info}

TAREAS:
1. Identifica los 3 insights más importantes sobre patrones de ventas
2. Detecta 2 riesgos potenciales en los datos o el negocio
3. Proporciona 3 recomendaciones específicas para optimizar ventas
4. Formula 1 pregunta estratégica para investigación futura

Formato tu respuesta con:
- **Insights Principales**: (lista numerada)
- **Riesgos Identificados**: (lista numerada)
- **Recomendaciones**: (lista numerada)
- **Pregunta Estratégica**: (una pregunta)
"""

            # Configurar la solicitud a Groq
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
                    st.markdown(insights)
                    
                    # Opción para descargar insights
                    st.download_button(
                        "Download Insights",
                        insights,
                        file_name="ai_insights.txt"
                    )
                    
                else:
                    st.error(f"Groq API Error: {response.text}")
                    
            except Exception as e:
                st.error(f"Request failed: {e}")