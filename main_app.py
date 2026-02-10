# =====================================================
# IMPORTS & CONFIG
# =====================================================
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import requests
import io
from datetime import datetime
import calendar

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
# CLEANING LOGIC (MEJORADA)
# =====================================================
def clean_data(df, remove_duplicates, impute_method, outlier_threshold):
    df_original = df.copy()
    
    # Registrar transformaciones
    transformations = []
    
    # Eliminar duplicados
    if remove_duplicates:
        duplicates_before = df.shape[0]
        df = df.drop_duplicates()
        duplicates_removed = duplicates_before - df.shape[0]
        if duplicates_removed > 0:
            transformations.append(f"📊 Se removieron {duplicates_removed} filas duplicadas")
    
    # Convertir fecha
    if "Transaction Date" in df.columns:
        df["Transaction Date"] = pd.to_datetime(df["Transaction Date"], errors="coerce")
        transformations.append("📅 'Transaction Date' convertida a datetime")
    
    # Limpieza especial para 'Discount Applied'
    if "Discount Applied" in df.columns:
        # Convertir a booleano manejando diferentes formatos
        discount_values = df["Discount Applied"].astype(str).str.lower().str.strip()
        
        # Mapeo de valores comunes a booleano
        discount_mapping = {
            'true': True, 'yes': True, '1': True, 'verdadero': True, 'si': True,
            'false': False, 'no': False, '0': False, 'falso': False
        }
        
        df["Discount Applied"] = discount_values.map(discount_mapping)
        null_before = df_original["Discount Applied"].isnull().sum()
        null_after = df["Discount Applied"].isnull().sum()
        
        if null_before > null_after:
            transformations.append(f"✅ 'Discount Applied' limpiado: {null_before - null_after} valores convertidos")
    
    # Imputación para columnas numéricas
    numeric_cols = df.select_dtypes(include=np.number).columns
    
    if len(numeric_cols) > 0 and impute_method != "Ninguna":
        missing_before = df[numeric_cols].isnull().sum().sum()
        
        if impute_method == "Media":
            df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].mean())
        elif impute_method == "Mediana":
            df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())
        elif impute_method == "Cero":
            df[numeric_cols] = df[numeric_cols].fillna(0)
        
        missing_after = df[numeric_cols].isnull().sum().sum()
        if missing_before > missing_after:
            transformations.append(f"🔧 Imputación ({impute_method}): {missing_before - missing_after} valores")
    
    # Manejo de outliers
    outliers_removed = 0
    for col in numeric_cols:
        mean = df[col].mean()
        std = df[col].std()
        if std > 0:  # Evitar división por cero
            upper = mean + outlier_threshold * std
            lower = mean - outlier_threshold * std
            outliers = ((df[col] > upper) | (df[col] < lower)).sum()
            outliers_removed += outliers
            df = df[(df[col] <= upper) & (df[col] >= lower)]
    
    if outliers_removed > 0:
        transformations.append(f"📈 Outliers removidos: {outliers_removed} (threshold: {outlier_threshold}σ)")
    
    return df, df_original, transformations

# =====================================================
# ANALYSIS FUNCTIONS
# =====================================================
def analyze_category_profitability(df):
    """Analiza rentabilidad por categoría"""
    if "Category" not in df.columns or "Total Spent" not in df.columns:
        return None
    
    # Calcular métricas por categoría
    category_analysis = df.groupby("Category").agg({
        "Total Spent": ["sum", "mean", "count"],
        "Quantity": "sum" if "Quantity" in df.columns else "count"
    }).round(2)
    
    # Aplanar columnas multi-index
    category_analysis.columns = ['_'.join(col).strip() for col in category_analysis.columns.values]
    
    # Renombrar para claridad
    category_analysis = category_analysis.rename(columns={
        "Total Spent_sum": "Ingreso_Total",
        "Total Spent_mean": "Ticket_Promedio",
        "Total Spent_count": "Transacciones",
        "Quantity_sum": "Cantidad_Total" if "Quantity" in df.columns else "Transacciones"
    })
    
    # Calcular porcentaje de contribución
    category_analysis["%_Contribución"] = (category_analysis["Ingreso_Total"] / 
                                         category_analysis["Ingreso_Total"].sum() * 100).round(2)
    
    # Ordenar por rentabilidad
    category_analysis = category_analysis.sort_values("Ingreso_Total", ascending=False)
    
    return category_analysis

def analyze_customer_segments(df):
    """Analiza segmentos de clientes"""
    segments = {}
    
    # Análisis por ubicación
    if "Location" in df.columns and "Total Spent" in df.columns:
        location_analysis = df.groupby("Location").agg({
            "Total Spent": ["sum", "mean", "count"],
            "Customer ID": "nunique" if "Customer ID" in df.columns else None
        }).round(2)
        
        if location_analysis.isnull().all().all():
            location_analysis = None
        else:
            segments["ubicacion"] = location_analysis
    
    # Análisis por método de pago
    if "Payment Method" in df.columns and "Total Spent" in df.columns:
        payment_analysis = df.groupby("Payment Method").agg({
            "Total Spent": ["sum", "mean", "count"]
        }).round(2)
        segments["metodo_pago"] = payment_analysis
    
    # Análisis por categoría preferida del cliente
    if "Customer ID" in df.columns and "Category" in df.columns:
        customer_category = df.groupby(["Customer ID", "Category"])["Total Spent"].sum().reset_index()
        top_category_per_customer = customer_category.loc[
            customer_category.groupby("Customer ID")["Total Spent"].idxmax()
        ]
        segments["categoria_preferida"] = top_category_per_customer["Category"].value_counts()
    
    return segments

def analyze_temporal_patterns(df):
    """Analiza patrones temporales"""
    if "Transaction Date" not in df.columns:
        return None
    
    patterns = {}
    
    # Crear columnas temporales
    df_temp = df.copy()
    df_temp["Year"] = df_temp["Transaction Date"].dt.year
    df_temp["Month"] = df_temp["Transaction Date"].dt.month
    df_temp["Month_Name"] = df_temp["Transaction Date"].dt.strftime('%B')
    df_temp["Day"] = df_temp["Transaction Date"].dt.day
    df_temp["Day_of_Week"] = df_temp["Transaction Date"].dt.dayofweek
    df_temp["Day_Name"] = df_temp["Transaction Date"].dt.strftime('%A')
    df_temp["Week"] = df_temp["Transaction Date"].dt.isocalendar().week
    df_temp["Quarter"] = df_temp["Transaction Date"].dt.quarter
    
    # Ventas por día de la semana
    weekday_sales = df_temp.groupby(["Day_Name", "Day_of_Week"])["Total Spent"].agg(['sum', 'mean', 'count']).reset_index()
    weekday_sales = weekday_sales.sort_values("Day_of_Week")
    patterns["dia_semana"] = weekday_sales
    
    # Ventas por mes
    monthly_sales = df_temp.groupby(["Month_Name", "Month"])["Total Spent"].agg(['sum', 'mean', 'count']).reset_index()
    monthly_sales = monthly_sales.sort_values("Month")
    patterns["mes"] = monthly_sales
    
    # Ventas por trimestre
    quarterly_sales = df_temp.groupby("Quarter")["Total Spent"].agg(['sum', 'mean', 'count']).reset_index()
    patterns["trimestre"] = quarterly_sales
    
    # Ventas por hora (si hubiera)
    if df_temp["Transaction Date"].dt.hour.nunique() > 1:
        df_temp["Hour"] = df_temp["Transaction Date"].dt.hour
        hourly_sales = df_temp.groupby("Hour")["Total Spent"].agg(['sum', 'mean', 'count']).reset_index()
        patterns["hora"] = hourly_sales
    
    return patterns

# =====================================================
# SIDEBAR NAVIGATION
# =====================================================
st.sidebar.title("📊 Navigation")
page = st.sidebar.radio(
    "Go to",
    ["ETL", "EDA", "Business Insights", "KPIs", "AI Insights"]
)

st.sidebar.title("📂 Data Source")

uploaded_file = st.sidebar.file_uploader("Upload CSV", type=['csv'])
url_input = st.sidebar.text_input("Or URL", placeholder="https://example.com/data.csv")

# =====================================================
# DATA LOADING
# =====================================================
df = None

try:
    if uploaded_file:
        df = load_file(uploaded_file)
        st.sidebar.success(f"✅ Loaded: {uploaded_file.name} ({len(df)} rows)")
    elif url_input and url_input.startswith(('http://', 'https://')):
        df = load_url(url_input)
        st.sidebar.success(f"✅ Loaded from URL ({len(df)} rows)")
except Exception as e:
    st.sidebar.error(f"❌ Error loading data: {str(e)}")

if df is None:
    st.info("👈 Please upload a CSV file or enter a URL to begin")
    st.stop()

# =====================================================
# ETL CONTROLS
# =====================================================
st.sidebar.title("🔧 ETL Controls")

remove_duplicates = st.sidebar.checkbox("Remove duplicates", value=True)

impute_method = st.sidebar.selectbox(
    "Imputation method",
    ["Ninguna", "Media", "Mediana", "Cero"],
    index=1
)

outlier_threshold = st.sidebar.slider("Outlier threshold (σ)", 1.0, 5.0, 3.0, 0.5)

df_clean, df_original, transformations = clean_data(df, remove_duplicates, impute_method, outlier_threshold)

# =====================================================
# GLOBAL FILTERS
# =====================================================
st.sidebar.title("🎛️ Global Filters")

if "Category" in df_clean.columns:
    categories = st.sidebar.multiselect(
        "Category",
        df_clean["Category"].dropna().unique(),
        default=df_clean["Category"].dropna().unique()
    )
    df_clean = df_clean[df_clean["Category"].isin(categories)]

if "Location" in df_clean.columns:
    locations = st.sidebar.multiselect(
        "Location",
        df_clean["Location"].dropna().unique(),
        default=df_clean["Location"].dropna().unique()
    )
    df_clean = df_clean[df_clean["Location"].isin(locations)]

# =====================================================
# ETL PAGE
# =====================================================
if page == "ETL":
    st.title("🔄 ETL Interactive Dashboard")
    
    # Mostrar transformaciones realizadas
    if transformations:
        st.subheader("Transformaciones aplicadas")
        for transform in transformations:
            st.write(f"• {transform}")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📋 Before Cleaning")
        st.metric("Rows", df_original.shape[0])
        st.metric("Columns", df_original.shape[1])
        st.dataframe(df_original.head(), use_container_width=True)
        
        # Estadísticas antes
        st.write("**Missing values before:**")
        st.dataframe(df_original.isnull().sum().to_frame("Count"), use_container_width=True)
    
    with col2:
        st.subheader("✨ After Cleaning")
        st.metric("Rows", df_clean.shape[0], delta=df_clean.shape[0]-df_original.shape[0])
        st.metric("Columns", df_clean.shape[1])
        st.dataframe(df_clean.head(), use_container_width=True)
        
        # Estadísticas después
        st.write("**Missing values after:**")
        st.dataframe(df_clean.isnull().sum().to_frame("Count"), use_container_width=True)
    
    # Descarga de datos limpios
    st.download_button(
        "💾 Download Clean CSV",
        df_clean.to_csv(index=False),
        "clean_data.csv",
        "text/csv"
    )

# =====================================================
# EDA PAGE
# =====================================================
elif page == "EDA":
    st.title("🔍 Exploratory Data Analysis")
    
    # Sección 1: Información general
    with st.expander("📊 Información General", expanded=True):
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Primeras filas")
            st.dataframe(df_clean.head())
        
        with col2:
            st.subheader("Resumen")
            buffer = io.StringIO()
            df_clean.info(buf=buffer)
            st.text(buffer.getvalue())
        
        st.subheader("Estadísticas descriptivas")
        st.dataframe(df_clean.describe())
    
    # Sección 2: Calidad de datos
    with st.expander("🧹 Calidad de Datos"):
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Valores Faltantes")
            missing_count = df_clean.isnull().sum()
            missing_percentage = (df_clean.isnull().sum() / len(df_clean)) * 100
            missing_data = pd.DataFrame({
                'Missing': missing_count,
                '%': missing_percentage.round(2)
            }).sort_values(by='Missing', ascending=False)
            
            missing_data = missing_data[missing_data['Missing'] > 0]
            if not missing_data.empty:
                st.dataframe(missing_data)
            else:
                st.success("✅ No hay valores faltantes")
        
        with col2:
            st.subheader("Filas Duplicadas")
            duplicate_rows_count = df_clean.duplicated().sum()
            if duplicate_rows_count > 0:
                st.warning(f"⚠️ {duplicate_rows_count} filas duplicadas encontradas")
                st.dataframe(df_clean[df_clean.duplicated()].head())
            else:
                st.success("✅ No hay filas duplicadas")
    
    # Sección 3: Visualizaciones
    with st.expander("📈 Visualizaciones"):
        numeric_cols = df_clean.select_dtypes(include=np.number).columns
        
        if len(numeric_cols) > 0:
            tab1, tab2, tab3 = st.tabs(["Univariado", "Bivariado", "Temporal"])
            
            with tab1:
                col = st.selectbox("Variable numérica", numeric_cols)
                fig = px.histogram(df_clean, x=col, title=f'Distribución de {col}')
                st.plotly_chart(fig, use_container_width=True)
            
            with tab2:
                if len(numeric_cols) > 1:
                    col1 = st.selectbox("Variable X", numeric_cols, key="x")
                    col2 = st.selectbox("Variable Y", numeric_cols, key="y")
                    
                    # Opción para agregar color por categoría
                    color_by = None
                    if "Category" in df_clean.columns:
                        color_by = st.checkbox("Color por categoría")
                    
                    if color_by:
                        fig = px.scatter(df_clean, x=col1, y=col2, color="Category",
                                       title=f'{col1} vs {col2} por Categoría')
                    else:
                        fig = px.scatter(df_clean, x=col1, y=col2, title=f'{col1} vs {col2}')
                    
                    st.plotly_chart(fig, use_container_width=True)
            
            with tab3:
                if "Transaction Date" in df_clean.columns:
                    df_temp = df_clean.copy()
                    df_temp = df_temp.sort_values("Transaction Date")
                    
                    # Opciones de agregación
                    col1, col2 = st.columns(2)
                    with col1:
                        freq = st.selectbox("Frecuencia", ["D", "W", "M", "Q", "Y"])
                    with col2:
                        metric = st.selectbox("Métrica", ["sum", "mean", "count"])
                    
                    # Resample
                    ts_data = df_temp.set_index("Transaction Date")["Total Spent"]
                    ts_resampled = ts_data.resample(freq).agg(metric).reset_index()
                    
                    fig = px.line(ts_resampled, x="Transaction Date", y="Total Spent",
                                title=f'Ventas Totales ({metric})')
                    st.plotly_chart(fig, use_container_width=True)

# =====================================================
# BUSINESS INSIGHTS PAGE
# =====================================================
elif page == "Business Insights":
    st.title("💡 Business Insights Dashboard")
    
    # Pregunta 1: Rentabilidad por categoría
    st.header("1. 📊 Análisis de Rentabilidad por Categoría")
    
    category_analysis = analyze_category_profitability(df_clean)
    
    if category_analysis is not None:
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Top 5 Categorías por Ingreso")
            top_categories = category_analysis.head().copy()
            st.dataframe(top_categories)
            
            # Gráfico de barras para top categorías
            fig = px.bar(top_categories, 
                        x=top_categories.index, 
                        y="Ingreso_Total",
                        title="Top 5 Categorías por Ingreso Total")
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.subheader("Bottom 5 Categorías por Ingreso")
            bottom_categories = category_analysis.tail().copy()
            st.dataframe(bottom_categories)
            
            # Gráfico de torta para distribución
            fig = px.pie(category_analysis, 
                        values="Ingreso_Total", 
                        names=category_analysis.index,
                        title="Distribución de Ingresos por Categoría")
            st.plotly_chart(fig, use_container_width=True)
        
        # Insights específicos
        st.subheader("🔑 Insights Estratégicos")
        top_category = category_analysis.iloc[0]
        bottom_category = category_analysis.iloc[-1]
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Categoría #1", 
                     top_category.name,
                     f"{top_category['%_Contribución']}% del total")
        
        with col2:
            st.metric("Ticket Promedio más alto", 
                     f"${category_analysis['Ticket_Promedio'].max():.2f}",
                     f"Categoría: {category_analysis['Ticket_Promedio'].idxmax()}")
        
        with col3:
            st.metric("Categoría menos rentable", 
                     bottom_category.name,
                     f"Solo {bottom_category['%_Contribución']}% del total")
    
    # Pregunta 2: Segmentos de clientes
    st.header("2. 👥 Análisis de Segmentos de Clientes")
    
    customer_segments = analyze_customer_segments(df_clean)
    
    if customer_segments:
        col1, col2 = st.columns(2)
        
        with col1:
            if "ubicacion" in customer_segments:
                st.subheader("📍 Por Ubicación")
                loc_data = customer_segments["ubicacion"]
                if isinstance(loc_data, pd.DataFrame):
                    # Aplanar columnas si es multi-index
                    if isinstance(loc_data.columns, pd.MultiIndex):
                        loc_data.columns = ['_'.join(col).strip() for col in loc_data.columns.values]
                    st.dataframe(loc_data)
                    
                    # Gráfico de ticket promedio por ubicación
                    if any("mean" in col for col in loc_data.columns):
                        mean_col = [col for col in loc_data.columns if "mean" in col][0]
                        fig = px.bar(loc_data, x=loc_data.index, y=mean_col,
                                   title="Ticket Promedio por Ubicación")
                        st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            if "metodo_pago" in customer_segments:
                st.subheader("💳 Por Método de Pago")
                pay_data = customer_segments["metodo_pago"]
                if isinstance(pay_data, pd.DataFrame):
                    # Aplanar columnas si es multi-index
                    if isinstance(pay_data.columns, pd.MultiIndex):
                        pay_data.columns = ['_'.join(col).strip() for col in pay_data.columns.values]
                    st.dataframe(pay_data)
                    
                    # Gráfico de distribución
                    if any("sum" in col for col in pay_data.columns):
                        sum_col = [col for col in pay_data.columns if "sum" in col][0]
                        fig = px.pie(pay_data, values=sum_col, names=pay_data.index,
                                   title="Distribución de Ventas por Método de Pago")
                        st.plotly_chart(fig, use_container_width=True)
        
        # Análisis de categoría preferida por cliente
        if "categoria_preferida" in customer_segments:
            st.subheader("🏷️ Categoría Preferida por Cliente")
            pref_data = customer_segments["categoria_preferida"]
            if isinstance(pref_data, pd.Series):
                st.dataframe(pref_data.head(10))
                
                fig = px.bar(pref_data.head(10), 
                           x=pref_data.head(10).index,
                           y=pref_data.head(10).values,
                           title="Top 10 Categorías Preferidas por Clientes")
                st.plotly_chart(fig, use_container_width=True)
    
    # Pregunta 3: Patrones temporales
    st.header("3. 📅 Análisis de Patrones Temporales")
    
    temporal_patterns = analyze_temporal_patterns(df_clean)
    
    if temporal_patterns:
        # Gráfico de ventas por día de la semana
        if "dia_semana" in temporal_patterns:
            st.subheader("📆 Ventas por Día de la Semana")
            weekday_data = temporal_patterns["dia_semana"]
            st.dataframe(weekday_data)
            
            fig = px.line(weekday_data, x="Day_Name", y="sum",
                         title="Ventas Totales por Día de la Semana")
            st.plotly_chart(fig, use_container_width=True)
        
        # Gráfico de ventas por mes
        if "mes" in temporal_patterns:
            st.subheader("🗓️ Ventas por Mes")
            monthly_data = temporal_patterns["mes"]
            st.dataframe(monthly_data)
            
            fig = px.line(monthly_data, x="Month_Name", y="sum",
                         title="Ventas Totales por Mes")
            st.plotly_chart(fig, use_container_width=True)
        
        # Heatmap de ventas (día de semana vs hora)
        if "Transaction Date" in df_clean.columns:
            st.subheader("🌡️ Heatmap de Ventas")
            df_temp = df_clean.copy()
            df_temp["Hour"] = df_temp["Transaction Date"].dt.hour
            df_temp["Weekday"] = df_temp["Transaction Date"].dt.dayofweek
            
            heatmap_data = df_temp.groupby(["Weekday", "Hour"])["Total Spent"].sum().unstack()
            
            fig = px.imshow(heatmap_data,
                          labels=dict(x="Hora del Día", y="Día de la Semana", color="Ventas"),
                          x=[f"{h}:00" for h in range(24)],
                          y=["Lun", "Mar", "Mié", "Jue", "Vie", "Sáb", "Dom"],
                          title="Heatmap de Ventas: Día vs Hora")
            st.plotly_chart(fig, use_container_width=True)
    
    # Recomendaciones estratégicas basadas en análisis
    st.header("🎯 Recomendaciones Estratégicas")
    
    rec_col1, rec_col2 = st.columns(2)
    
    with rec_col1:
        st.subheader("📦 Para Inventario")
        st.write("""
        1. **Enfocar stock** en categorías de alto ingreso
        2. **Reducir inventario** de categorías de baja rentabilidad
        3. **Negociar mejores términos** con proveedores de categorías top
        4. **Considerar bundling** de productos de alta y baja rentabilidad
        """)
    
    with rec_col2:
        st.subheader("👥 Para Marketing")
        st.write("""
        1. **Campañas personalizadas** para segmentos de alto ticket
        2. **Programas de lealtad** para clientes de ubicaciones específicas
        3. **Promociones estratégicas** en días/horas de baja venta
        4. **Upselling cruzado** basado en categorías preferidas
        """)

# =====================================================
# KPIs PAGE
# =====================================================
elif page == "KPIs":
    st.title("📊 KPIs Dashboard")
    
    # KPIs principales
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        total_sales = df_clean['Total Spent'].sum()
        st.metric("💰 Total Sales", f"${total_sales:,.0f}")
    
    with col2:
        total_transactions = len(df_clean)
        st.metric("🛒 Transactions", f"{total_transactions:,}")
    
    with col3:
        avg_ticket = df_clean['Total Spent'].mean()
        st.metric("🎫 Average Ticket", f"${avg_ticket:,.2f}")
    
    with col4:
        unique_customers = df_clean['Customer ID'].nunique() if 'Customer ID' in df_clean.columns else "N/A"
        st.metric("👥 Unique Customers", f"{unique_customers}")
    
    # KPIs secundarios
    st.subheader("📈 Detailed Metrics")
    
    col1, col2 = st.columns(2)
    
    with col1:
        if "Quantity" in df_clean.columns and "Total Spent" in df_clean.columns:
            total_quantity = df_clean['Quantity'].sum()
            if total_quantity > 0:
                avg_price = df_clean['Total Spent'].sum() / total_quantity
                st.metric("🏷️ Average Price per Unit", f"${avg_price:.2f}")
        
        if "Discount Applied" in df_clean.columns:
            discount_rate = df_clean['Discount Applied'].mean() * 100 if df_clean['Discount Applied'].dtype == bool else 0
            st.metric("🎁 Discount Rate", f"{discount_rate:.1f}%")
    
    with col2:
        if "Payment Method" in df_clean.columns:
            payment_dist = df_clean['Payment Method'].value_counts()
            st.write("💳 Payment Method Distribution:")
            st.dataframe(payment_dist)
        
        if "Location" in df_clean.columns:
            location_dist = df_clean['Location'].value_counts()
            st.write("📍 Sales by Location:")
            st.dataframe(location_dist)
    
    # KPIs por categoría
    if "Category" in df_clean.columns:
        st.subheader("🏷️ KPIs por Categoría")
        
        category_kpis = df_clean.groupby("Category").agg({
            "Total Spent": ["sum", "mean", "count"],
            "Customer ID": "nunique" if "Customer ID" in df_clean.columns else None
        }).round(2)
        
        st.dataframe(category_kpis)

# =====================================================
# AI INSIGHTS PAGE
# =====================================================
elif page == "AI Insights":
    st.title("🤖 AI Generated Insights")
    
    api_key = st.sidebar.text_input("Groq API Key", type="password")
    
    if st.button("🚀 Generate Insights"):
        if not api_key:
            st.warning("⚠️ Please enter your Groq API Key")
        else:
            # Preparar resumen de datos para el prompt
            data_summary = {
                "shape": df_clean.shape,
                "columns": list(df_clean.columns),
                "numeric_summary": df_clean.describe().to_string(),
                "categorical_summary": df_clean.select_dtypes(include=['object']).describe().to_string() if not df_clean.select_dtypes(include=['object']).empty else "No categorical columns",
                "missing_values": df_clean.isnull().sum().sum()
            }
            
            # Análisis de categorías para incluir en el prompt
            category_insights = ""
            if "Category" in df_clean.columns:
                cat_analysis = analyze_category_profitability(df_clean)
                if cat_analysis is not None:
                    category_insights = f"""
ANÁLISIS DE CATEGORÍAS:
{cat_analysis.to_string()}

TOP 3 CATEGORÍAS:
1. {cat_analysis.index[0]}: ${cat_analysis.iloc[0]['Ingreso_Total']:,.2f} ({cat_analysis.iloc[0]['%_Contribución']}%)
2. {cat_analysis.index[1]}: ${cat_analysis.iloc[1]['Ingreso_Total']:,.2f} ({cat_analysis.iloc[1]['%_Contribución']}%)
3. {cat_analysis.index[2]}: ${cat_analysis.iloc[2]['Ingreso_Total']:,.2f} ({cat_analysis.iloc[2]['%_Contribución']}%)
"""
            
            prompt = f"""
Eres un analista de datos senior especializado en retail. Analiza los siguientes datos y proporciona insights estratégicos:

CONTEXTO DEL DATASET:
- Filas: {data_summary['shape'][0]}
- Columnas: {data_summary['shape'][1]}
- Columnas disponibles: {', '.join(data_summary['columns'])}
- Valores faltantes: {data_summary['missing_values']}

RESUMEN NUMÉRICO:
{data_summary['numeric_summary']}

{category_insights}

TAREAS ESPECÍFICAS:
1. Identifica los 3 insights más importantes sobre patrones de ventas
2. Detecta 2 riesgos potenciales en los datos o el negocio
3. Proporciona 3 recomendaciones específicas para optimizar ventas
4. Formula 1 pregunta estratégica para investigación futura
5. Sugiere 2 acciones inmediatas basadas en los datos

Formato tu respuesta en español con:
- **🔍 Insights Principales**: (lista numerada)
- **⚠️ Riesgos Identificados**: (lista numerada)
- **🎯 Recomendaciones**: (lista numerada)
- **🤔 Pregunta Estratégica**: (una pregunta)
- **🚀 Acciones Inmediatas**: (lista numerada)
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
                "temperature": 0.3,
                "max_tokens": 1000
            }
            
            try:
                with st.spinner("🧠 Generating AI Insights..."):
                    response = requests.post(
                        "https://api.groq.com/openai/v1/chat/completions",
                        headers=headers,
                        json=payload,
                        timeout=120
                    )
                
                if response.status_code == 200:
                    result = response.json()
                    insights = result["choices"][0]["message"]["content"]
                    
                    st.subheader("💡 AI Strategic Insights")
                    st.markdown(insights)
                    
                    # Opción para descargar insights
                    st.download_button(
                        "📥 Download Insights",
                        insights,
                        file_name=f"ai_insights_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
                    )
                    
                else:
                    st.error(f"❌ Groq API Error: {response.text}")
                    
            except Exception as e:
                st.error(f"❌ Request failed: {str(e)}")
                st.info("💡 Tip: Check your API key and internet connection")