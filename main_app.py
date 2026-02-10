# =====================================================
# IMPORTS & CONFIG
# =====================================================
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import io
import warnings
warnings.filterwarnings('ignore')

# Configuración de página
st.set_page_config(
    page_title="Retail Sales Dashboard", 
    layout="wide",
    page_icon="📊"
)

# =====================================================
# CACHE FUNCTIONS
# =====================================================
@st.cache_data
def load_file(file):
    """Carga archivo CSV"""
    return pd.read_csv(file)

# =====================================================
# ETL MEJORADO - MÁS CONSERVADOR
# =====================================================
def enhanced_clean_data(df, remove_duplicates=True, impute_method="Mediana", outlier_threshold=1.5):
    """
    Función de limpieza CONSERVADORA - elimina más datos en lugar de imputar
    """
    transformations = []
    df_original = df.copy()
    
    # 1. ANÁLISIS INICIAL
    initial_rows = len(df)
    transformations.append(f"📊 **ANÁLISIS INICIAL:** {initial_rows:,} registros, {len(df.columns)} columnas")
    
    # 2. ELIMINAR REGISTROS CON DEMASIADOS VALORES FALTANTES (CONSERVADOR)
    transformations.append("\n🗑️ **ELIMINACIÓN DE REGISTROS PROBLEMÁTICOS:**")
    
    # Eliminar registros donde falten datos críticos
    critical_cols = ['Transaction Date', 'Total Spent', 'Category']
    for col in critical_cols:
        if col in df.columns:
            missing_count = df[col].isnull().sum()
            if missing_count > 0:
                df = df.dropna(subset=[col])
                transformations.append(f"   • Eliminados {missing_count:,} registros sin '{col}'")
    
    # 3. CONVERSIONES DE TIPO DE DATOS
    transformations.append("\n🔄 **CONVERSIÓN DE TIPOS:**")
    
    # Fechas - eliminar inválidas
    if 'Transaction Date' in df.columns:
        df['Transaction Date'] = pd.to_datetime(df['Transaction Date'], errors='coerce')
        invalid_dates = df['Transaction Date'].isnull().sum()
        if invalid_dates > 0:
            df = df.dropna(subset=['Transaction Date'])
            transformations.append(f"   • Eliminadas {invalid_dates:,} fechas inválidas")
    
    # Columnas numéricas - mantener solo valores válidos
    numeric_cols = ['Price Per Unit', 'Quantity', 'Total Spent']
    for col in numeric_cols:
        if col in df.columns:
            # Convertir a numérico y eliminar inválidos
            df[col] = pd.to_numeric(df[col], errors='coerce')
            invalid_count = df[col].isnull().sum() - df_original[col].isnull().sum()
            if invalid_count > 0:
                transformations.append(f"   • {col}: {invalid_count:,} valores no numéricos eliminados")
    
    # 4. ELIMINAR DUPLICADOS
    if remove_duplicates:
        duplicates = df.duplicated().sum()
        if duplicates > 0:
            df = df.drop_duplicates()
            transformations.append(f"\n🧹 **ELIMINADOS {duplicates:,} REGISTROS DUPLICADOS**")
    
    # 5. IMPUTACIÓN MUY CONSERVADORA - SOLO PARA COLUMNAS CRÍTICAS
    transformations.append("\n🔧 **IMPUTACIÓN LIMITADA:**")
    
    # Solo imputar valores numéricos con mediana (más robusta que la media)
    for col in ['Total Spent', 'Quantity', 'Price Per Unit']:
        if col in df.columns:
            missing_before = df[col].isnull().sum()
            if missing_before > 0 and impute_method in ["Mediana", "Media"]:
                if impute_method == "Mediana":
                    fill_value = df[col].median()
                else:
                    fill_value = df[col].mean()
                
                df[col] = df[col].fillna(fill_value)
                transformations.append(f"   • {col}: {missing_before:,} valores imputados con {impute_method.lower()}")
    
    # 6. ELIMINAR OUTLIERS EXTREMOS
    transformations.append("\n📈 **ELIMINACIÓN DE OUTLIERS:**")
    
    for col in ['Total Spent', 'Quantity', 'Price Per Unit']:
        if col in df.columns and df[col].notnull().any():
            # Método más conservador: eliminar solo valores extremos
            Q1 = df[col].quantile(0.01)  # Percentil 1
            Q3 = df[col].quantile(0.99)  # Percentil 99
            IQR = Q3 - Q1
            
            lower_bound = Q1 - outlier_threshold * IQR
            upper_bound = Q3 + outlier_threshold * IQR
            
            outliers = ((df[col] < lower_bound) | (df[col] > upper_bound)).sum()
            if outliers > 0:
                df = df[(df[col] >= lower_bound) & (df[col] <= upper_bound)]
                transformations.append(f"   • {col}: {outliers:,} outliers extremos eliminados")
    
    # 7. VALIDACIONES FINALES
    transformations.append("\n✅ **VALIDACIONES FINALES:**")
    
    # Asegurar valores positivos
    for col in ['Total Spent', 'Quantity']:
        if col in df.columns:
            negative_count = (df[col] < 0).sum()
            if negative_count > 0:
                df.loc[df[col] < 0, col] = 0
                transformations.append(f"   • {col}: {negative_count:,} valores negativos corregidos a 0")
    
    # 8. RESULTADOS FINALES
    final_rows = len(df)
    rows_removed = initial_rows - final_rows
    missing_after = df.isnull().sum().sum()
    
    transformations.append(f"\n🎯 **RESULTADO FINAL:**")
    transformations.append(f"   • Registros finales: {final_rows:,}")
    transformations.append(f"   • Registros eliminados: {rows_removed:,} ({rows_removed/initial_rows*100:.1f}%)")
    transformations.append(f"   • Valores faltantes restantes: {missing_after:,}")
    
    if missing_after > 0:
        remaining = df.isnull().sum()
        remaining = remaining[remaining > 0]
        for col, count in remaining.items():
            transformations.append(f"   • {col}: {count:,} faltantes ({count/final_rows*100:.1f}%)")
    
    return df, df_original, transformations

# =====================================================
# ANÁLISIS SIMPLIFICADO
# =====================================================
def analyze_category_profitability(df):
    """Análisis simple por categoría"""
    if "Category" not in df.columns or "Total Spent" not in df.columns:
        return None
    
    analysis = df.groupby("Category").agg({
        "Total Spent": "sum",
        "Transaction ID": "count"
    }).rename(columns={"Total Spent": "Ingreso_Total", "Transaction ID": "Transacciones"})
    
    analysis["%_Contribucion"] = (analysis["Ingreso_Total"] / analysis["Ingreso_Total"].sum() * 100).round(2)
    analysis["Ticket_Promedio"] = (analysis["Ingreso_Total"] / analysis["Transacciones"]).round(2)
    
    return analysis.sort_values("Ingreso_Total", ascending=False)

def analyze_daily_patterns(df):
    """Análisis simple por día"""
    if "Transaction Date" not in df.columns:
        return None
    
    df_temp = df.copy()
    df_temp["Dia_Semana"] = df_temp["Transaction Date"].dt.day_name()
    df_temp["Mes"] = df_temp["Transaction Date"].dt.month_name()
    
    patterns = {}
    
    # Por día de semana
    daily = df_temp.groupby("Dia_Semana").agg({
        "Total Spent": "sum",
        "Transaction ID": "count"
    }).rename(columns={"Total Spent": "Ventas_Totales", "Transaction ID": "Transacciones"})
    
    # Ordenar días
    day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    daily = daily.reindex(day_order)
    patterns["dia_semana"] = daily
    
    # Por mes
    monthly = df_temp.groupby("Mes").agg({
        "Total Spent": "sum",
        "Transaction ID": "count"
    }).rename(columns={"Total Spent": "Ventas_Totales", "Transaction ID": "Transacciones"})
    
    # Ordenar meses
    month_order = ['January', 'February', 'March', 'April', 'May', 'June', 
                   'July', 'August', 'September', 'October', 'November', 'December']
    monthly = monthly.reindex([m for m in month_order if m in monthly.index])
    patterns["mes"] = monthly
    
    return patterns

# =====================================================
# INTERFAZ PRINCIPAL
# =====================================================
# Sidebar principal
st.sidebar.title("📊 Retail Sales Dashboard")

# Navegación
page = st.sidebar.radio(
    "Navegación",
    ["🏠 Inicio", "🔄 Limpieza", "📊 Análisis", "📈 KPIs"]
)

# Carga de datos
st.sidebar.title("📂 Carga de Datos")
uploaded_file = st.sidebar.file_uploader("Subir archivo CSV", type=['csv'])

# Configuración ETL
st.sidebar.title("⚙️ Configuración")
with st.sidebar.expander("Opciones de limpieza"):
    remove_duplicates = st.checkbox("Eliminar duplicados", value=True)
    impute_method = st.selectbox(
        "Método de imputación",
        ["Mediana", "Media", "Ninguno"],
        index=0,
        help="La mediana es más robusta contra outliers"
    )
    outlier_threshold = st.slider("Umbral outliers", 1.0, 3.0, 1.5, 0.1)

# =====================================================
# CARGA Y PROCESAMIENTO DE DATOS
# =====================================================
df = None
df_clean = None

if uploaded_file is not None:
    try:
        df = load_file(uploaded_file)
        st.sidebar.success(f"✅ Cargado: {uploaded_file.name}")
        
        # Aplicar ETL
        df_clean, df_original, transformations = enhanced_clean_data(
            df, 
            remove_duplicates, 
            impute_method,
            outlier_threshold
        )
        
    except Exception as e:
        st.sidebar.error(f"❌ Error: {str(e)}")

# =====================================================
# PÁGINAS
# =====================================================
if df is None:
    st.title("📊 Retail Sales Dashboard")
    st.markdown("---")
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.markdown("""
        ## Bienvenido
        
        ### Funcionalidades:
        ✅ **Carga y limpieza** de datos  
        ✅ **Análisis** simplificado  
        ✅ **Visualización** de KPIs  
        
        ### Para comenzar:
        1. Sube un archivo CSV
        2. Configura las opciones
        3. Navega entre secciones
        """)
    
    with col2:
        st.info("👈 Usa el sidebar para cargar tus datos")

elif page == "🏠 Inicio":
    st.title("🏠 Panel de Control")
    
    if df_clean is not None:
        # Métricas principales
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            total_sales = df_clean["Total Spent"].sum() if "Total Spent" in df_clean.columns else 0
            st.metric("💰 Ventas Totales", f"${total_sales:,.0f}")
        
        with col2:
            total_transactions = len(df_clean)
            st.metric("🛒 Transacciones", f"{total_transactions:,}")
        
        with col3:
            avg_ticket = df_clean["Total Spent"].mean() if "Total Spent" in df_clean.columns else 0
            st.metric("🎫 Ticket Promedio", f"${avg_ticket:,.2f}")
        
        with col4:
            unique_categories = df_clean["Category"].nunique() if "Category" in df_clean.columns else 0
            st.metric("🏷️ Categorías", f"{unique_categories}")
        
        # Resumen de limpieza
        st.markdown("---")
        st.subheader("📋 Resumen de Limpieza")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.info(f"**Registros originales:** {len(df_original):,}")
            st.info(f"**Registros limpios:** {len(df_clean):,}")
            st.info(f"**Reducción:** {len(df_original) - len(df_clean):,} registros")
        
        with col2:
            missing_before = df_original.isnull().sum().sum()
            missing_after = df_clean.isnull().sum().sum()
            st.warning(f"**Valores faltantes antes:** {missing_before:,}")
            st.success(f"**Valores faltantes después:** {missing_after:,}")
        
        # Muestra de datos
        st.markdown("---")
        st.subheader("📄 Muestra de Datos")
        
        tab1, tab2 = st.tabs(["Primeros registros", "Estadísticas"])
        
        with tab1:
            st.dataframe(df_clean.head(), use_container_width=True)
        
        with tab2:
            numeric_cols = df_clean.select_dtypes(include=[np.number]).columns
            if len(numeric_cols) > 0:
                st.dataframe(df_clean[numeric_cols].describe(), use_container_width=True)

elif page == "🔄 Limpieza":
    st.title("🔄 Proceso de Limpieza")
    
    if df_clean is not None:
        # Transformaciones aplicadas
        st.subheader("📋 Transformaciones Aplicadas")
        
        with st.expander("Ver detalles", expanded=True):
            for transform in transformations:
                st.write(transform)
        
        # Comparación visual
        st.markdown("---")
        st.subheader("📊 Comparación Visual")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Antes de la limpieza**")
            st.dataframe(df_original.head(), use_container_width=True)
            
            # Valores faltantes antes
            missing_before = df_original.isnull().sum()
            if missing_before.sum() > 0:
                st.write("**Valores faltantes (antes):**")
                st.dataframe(missing_before[missing_before > 0].to_frame("Cantidad"), use_container_width=True)
        
        with col2:
            st.write("**Después de la limpieza**")
            st.dataframe(df_clean.head(), use_container_width=True)
            
            # Valores faltantes después
            missing_after = df_clean.isnull().sum()
            if missing_after.sum() > 0:
                st.write("**Valores faltantes (después):**")
                st.dataframe(missing_after[missing_after > 0].to_frame("Cantidad"), use_container_width=True)
        
        # Descarga de datos
        st.markdown("---")
        st.subheader("💾 Exportar Datos")
        
        col1, col2 = st.columns(2)
        
        with col1:
            csv = df_clean.to_csv(index=False)
            st.download_button(
                "📥 Descargar CSV",
                csv,
                "datos_limpiados.csv",
                "text/csv"
            )
        
        with col2:
            json_str = df_clean.to_json(orient='records', indent=2)
            st.download_button(
                "📥 Descargar JSON",
                json_str,
                "datos_limpiados.json",
                "application/json"
            )

elif page == "📊 Análisis":
    st.title("📊 Análisis de Datos")
    
    if df_clean is not None:
        # Análisis por categoría
        st.header("🏷️ Análisis por Categoría")
        
        category_analysis = analyze_category_profitability(df_clean)
        
        if category_analysis is not None:
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Top Categorías")
                st.dataframe(
                    category_analysis.head(10),
                    use_container_width=True
                )
            
            with col2:
                # Gráfico de torta para top 5
                top5 = category_analysis.head(5)
                fig = px.pie(
                    top5,
                    values="Ingreso_Total",
                    names=top5.index,
                    title="Top 5 Categorías por Ingreso"
                )
                st.plotly_chart(fig, use_container_width=True)
            
            # Gráfico de barras
            fig = px.bar(
                category_analysis.head(10),
                x=category_analysis.head(10).index,
                y="Ingreso_Total",
                title="Ingreso Total por Categoría (Top 10)"
            )
            st.plotly_chart(fig, use_container_width=True)
        
        # Análisis temporal
        st.header("📅 Análisis Temporal")
        
        temporal_patterns = analyze_daily_patterns(df_clean)
        
        if temporal_patterns:
            col1, col2 = st.columns(2)
            
            with col1:
                if "dia_semana" in temporal_patterns:
                    daily_data = temporal_patterns["dia_semana"]
                    
                    # Gráfico de barras CORREGIDO
                    fig = px.bar(
                        daily_data.reset_index(),
                        x="Dia_Semana",
                        y="Ventas_Totales",
                        title="Ventas por Día de la Semana"
                    )
                    st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                if "mes" in temporal_patterns:
                    monthly_data = temporal_patterns["mes"]
                    
                    # Gráfico de líneas
                    fig = px.line(
                        monthly_data.reset_index(),
                        x="Mes",
                        y="Ventas_Totales",
                        title="Ventas por Mes",
                        markers=True
                    )
                    st.plotly_chart(fig, use_container_width=True)
        
        # Distribución de ventas
        st.header("📈 Distribución de Ventas")
        
        if "Total Spent" in df_clean.columns:
            col1, col2 = st.columns(2)
            
            with col1:
                # Histograma
                fig = px.histogram(
                    df_clean,
                    x="Total Spent",
                    nbins=50,
                    title="Distribución de Montos de Venta",
                    labels={"Total Spent": "Monto ($)"}
                )
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                # Box plot
                fig = px.box(
                    df_clean,
                    y="Total Spent",
                    title="Distribución de Montos (Box Plot)",
                    labels={"Total Spent": "Monto ($)"}
                )
                st.plotly_chart(fig, use_container_width=True)

elif page == "📈 KPIs":
    st.title("📈 Panel de KPIs")
    
    if df_clean is not None:
        # KPIs principales
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            total_sales = df_clean["Total Spent"].sum() if "Total Spent" in df_clean.columns else 0
            st.metric("💰 Ventas Totales", f"${total_sales:,.0f}")
        
        with col2:
            avg_ticket = df_clean["Total Spent"].mean() if "Total Spent" in df_clean.columns else 0
            st.metric("🎫 Ticket Promedio", f"${avg_ticket:,.2f}")
        
        with col3:
            total_transactions = len(df_clean)
            st.metric("🛒 Transacciones", f"{total_transactions:,}")
        
        with col4:
            if "Quantity" in df_clean.columns:
                total_quantity = df_clean["Quantity"].sum()
                st.metric("📦 Unidades Vendidas", f"{total_quantity:,}")
        
        # KPIs por categoría
        st.markdown("---")
        st.subheader("🏷️ KPIs por Categoría")
        
        if "Category" in df_clean.columns:
            category_kpis = df_clean.groupby("Category").agg({
                "Total Spent": ["sum", "mean", "count"],
                "Quantity": "sum" if "Quantity" in df_clean.columns else None
            }).round(2)
            
            # Aplanar columnas multi-index
            category_kpis.columns = ['_'.join(col).strip('_') for col in category_kpis.columns.values]
            st.dataframe(category_kpis, use_container_width=True)
            
            # Gráfico de contribución
            category_sales = df_clean.groupby("Category")["Total Spent"].sum().sort_values(ascending=False).head(10)
            fig = px.bar(
                category_sales,
                x=category_sales.values,
                y=category_sales.index,
                orientation='h',
                title="Top 10 Categorías por Ventas"
            )
            st.plotly_chart(fig, use_container_width=True)
        
        # KPIs por ubicación
        st.markdown("---")
        st.subheader("📍 KPIs por Ubicación")
        
        if "Location" in df_clean.columns:
            location_kpis = df_clean.groupby("Location").agg({
                "Total Spent": ["sum", "mean", "count"],
                "Customer ID": "nunique" if "Customer ID" in df_clean.columns else None
            }).round(2)
            
            if not location_kpis.empty:
                st.dataframe(location_kpis, use_container_width=True)
                
                # Gráfico de ventas por ubicación
                location_sales = df_clean.groupby("Location")["Total Spent"].sum()
                fig = px.bar(
                    location_sales,
                    x=location_sales.index,
                    y=location_sales.values,
                    title="Ventas por Ubicación"
                )
                st.plotly_chart(fig, use_container_width=True)