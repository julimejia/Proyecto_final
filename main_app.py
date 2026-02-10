# =====================================================
# IMPORTS & CONFIGURACIÓN
# =====================================================
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Configuración de página
st.set_page_config(
    page_title="Dashboard Retail Inteligente", 
    layout="wide",
    page_icon="📊"
)

# Estilos CSS personalizados
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1E3A8A;
        font-weight: bold;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.8rem;
        color: #3B82F6;
        font-weight: 600;
        margin-bottom: 0.5rem;
    }
    .metric-card {
        background-color: #F3F4F6;
        padding: 1rem;
        border-radius: 10px;
        border-left: 5px solid #1E3A8A;
        margin-bottom: 1rem;
    }
    .stButton>button {
        background-color: #1E3A8A;
        color: white;
        font-weight: bold;
        border: none;
        padding: 0.5rem 1rem;
        border-radius: 5px;
    }
    .stButton>button:hover {
        background-color: #3B82F6;
    }
</style>
""", unsafe_allow_html=True)


# =====================================================
# FUNCIONES DE CACHÉ
# =====================================================
@st.cache_data
def load_file(file):
    """Carga archivo CSV con caché"""
    try:
        df = pd.read_csv(file)
        return df
    except Exception as e:
        st.error(f"Error al cargar archivo: {str(e)}")
        return None


# =====================================================
# LIMPIEZA DE DATOS (SIMPLIFICADA Y EFECTIVA)
# =====================================================
def clean_retail_data(df):
    """
    Limpieza de datos retail basada en el enfoque del notebook
    """
    transformations = []
    df_original = df.copy()
    
    # 1. ANÁLISIS INICIAL
    initial_rows = len(df)
    initial_cols = len(df.columns)
    transformations.append(f"📊 **ANÁLISIS INICIAL:** {initial_rows:,} registros, {initial_cols} columnas")
    
    # 2. ELIMINAR COLUMNAS INNECESARIAS
    columns_to_drop = []
    if 'Transaction ID' in df.columns:
        columns_to_drop.append('Transaction ID')
    if 'Item' in df.columns:
        columns_to_drop.append('Item')
    
    if columns_to_drop:
        df = df.drop(columns=columns_to_drop)
        transformations.append(f"🗑️ **Columnas eliminadas:** {', '.join(columns_to_drop)}")
    
    # 3. NORMALIZAR NOMBRES DE COLUMNAS
    df.columns = df.columns.str.lower().str.replace(' ', '_', regex=True)
    transformations.append("🔄 **Columnas convertidas a snake_case**")
    
    # 4. MANEJO DE VALORES FALTANTES
    transformations.append("\n🔍 **MANEJO DE VALORES FALTANTES:**")
    
    # Rellenar discount_applied con 0
    if 'discount_applied' in df.columns:
        missing_discount = df['discount_applied'].isnull().sum()
        df['discount_applied'] = df['discount_applied'].fillna(0).astype(int)
        transformations.append(f"   • discount_applied: {missing_discount:,} valores nulos rellenados con 0")
    
    # Eliminar filas con otros valores nulos
    missing_before = df.isnull().sum().sum()
    df = df.dropna()
    missing_after = df.isnull().sum().sum()
    rows_removed = missing_before - missing_after
    transformations.append(f"   • Eliminadas {rows_removed:,} filas con valores nulos")
    
    # 5. CONVERSIÓN DE TIPOS DE DATOS
    transformations.append("\n🔄 **CONVERSIÓN DE TIPOS DE DATOS:**")
    
    # Convertir columnas categóricas
    categorical_cols = ['category', 'payment_method', 'location']
    for col in categorical_cols:
        if col in df.columns:
            df[col] = df[col].astype('category')
            transformations.append(f"   • {col}: convertida a category")
    
    # Convertir fecha
    if 'transaction_date' in df.columns:
        df['transaction_date'] = pd.to_datetime(df['transaction_date'], errors='coerce')
        # Eliminar fechas inválidas
        invalid_dates = df['transaction_date'].isnull().sum()
        if invalid_dates > 0:
            df = df.dropna(subset=['transaction_date'])
            transformations.append(f"   • transaction_date: {invalid_dates:,} fechas inválidas eliminadas")
    
    # Asegurar tipos numéricos
    numeric_cols = ['quantity', 'price_per_unit', 'total_spent']
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # 6. FEATURE ENGINEERING BÁSICO
    transformations.append("\n⚙️ **FEATURE ENGINEERING:**")
    
    if 'transaction_date' in df.columns:
        # Extraer características temporales
        df['year'] = df['transaction_date'].dt.year
        df['month'] = df['transaction_date'].dt.month
        df['day'] = df['transaction_date'].dt.day
        df['weekday'] = df['transaction_date'].dt.day_name()
        df['month_name'] = df['transaction_date'].dt.month_name()
        transformations.append("   • Características temporales extraídas (año, mes, día, día de semana)")
    
    # 7. VALIDACIONES FINALES
    final_rows = len(df)
    rows_removed_total = initial_rows - final_rows
    
    transformations.append(f"\n✅ **RESULTADO FINAL:**")
    transformations.append(f"   • Registros finales: {final_rows:,}")
    transformations.append(f"   • Registros eliminados: {rows_removed_total:,} ({rows_removed_total/initial_rows*100:.1f}%)")
    transformations.append(f"   • Valores faltantes restantes: {df.isnull().sum().sum():,}")
    
    return df, df_original, transformations


# =====================================================
# ANÁLISIS PARA PREGUNTAS DE NEGOCIO
# =====================================================
def analyze_category_profitability(df):
    """Analiza rentabilidad por categoría (Pregunta 1)"""
    if "category" not in df.columns or "total_spent" not in df.columns:
        return None
    
    analysis = df.groupby("category").agg({
        "total_spent": ["sum", "mean", "count"]
    }).round(2)
    
    # Aplanar columnas multi-index
    analysis.columns = ['_'.join(col).strip('_') for col in analysis.columns.values]
    
    # Renombrar columnas para claridad
    analysis = analysis.rename(columns={
        'total_spent_sum': 'ingreso_total',
        'total_spent_mean': 'ticket_promedio',
        'total_spent_count': 'transacciones'
    })
    
    # Calcular rentabilidad (ingreso por transacción)
    analysis['rentabilidad'] = (analysis['ingreso_total'] / analysis['transacciones']).round(2)
    
    return analysis.sort_values('ingreso_total', ascending=False)


def analyze_customer_segments(df):
    """Analiza segmentos de clientes (Pregunta 2)"""
    results = {}
    
    # Análisis por ubicación
    if 'location' in df.columns and 'total_spent' in df.columns:
        location_analysis = df.groupby('location').agg({
            'total_spent': ['mean', 'sum', 'count']
        }).round(2)
        
        if location_analysis.columns.nlevels > 1:
            location_analysis.columns = ['_'.join(col).strip('_') for col in location_analysis.columns.values]
        
        results['ubicacion'] = location_analysis.sort_values('total_spent_mean', ascending=False)
    
    # Análisis por método de pago
    if 'payment_method' in df.columns and 'total_spent' in df.columns:
        payment_analysis = df.groupby('payment_method').agg({
            'total_spent': ['mean', 'sum', 'count']
        }).round(2)
        
        if payment_analysis.columns.nlevels > 1:
            payment_analysis.columns = ['_'.join(col).strip('_') for col in payment_analysis.columns.values]
        
        results['metodo_pago'] = payment_analysis.sort_values('total_spent_mean', ascending=False)
    
    # Análisis por categoría comprada
    if 'category' in df.columns and 'total_spent' in df.columns:
        category_analysis = df.groupby('category').agg({
            'total_spent': ['mean', 'sum', 'count']
        }).round(2)
        
        if category_analysis.columns.nlevels > 1:
            category_analysis.columns = ['_'.join(col).strip('_') for col in category_analysis.columns.values]
        
        results['categoria'] = category_analysis.sort_values('total_spent_mean', ascending=False)
    
    return results


def analyze_temporal_patterns(df):
    """Analiza patrones temporales (Pregunta 3)"""
    results = {}
    
    if 'transaction_date' in df.columns and 'total_spent' in df.columns:
        # Análisis por día de la semana
        if 'weekday' in df.columns:
            daily_pattern = df.groupby('weekday').agg({
                'total_spent': ['sum', 'mean', 'count']
            }).round(2)
            
            # Ordenar días
            day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
            daily_pattern = daily_pattern.reindex(day_order)
            
            if daily_pattern.columns.nlevels > 1:
                daily_pattern.columns = ['_'.join(col).strip('_') for col in daily_pattern.columns.values]
            
            results['dia_semana'] = daily_pattern
        
        # Análisis por mes
        if 'month_name' in df.columns:
            monthly_pattern = df.groupby('month_name').agg({
                'total_spent': ['sum', 'mean', 'count']
            }).round(2)
            
            # Ordenar meses
            month_order = ['January', 'February', 'March', 'April', 'May', 'June', 
                         'July', 'August', 'September', 'October', 'November', 'December']
            monthly_pattern = monthly_pattern.reindex([m for m in month_order if m in monthly_pattern.index])
            
            if monthly_pattern.columns.nlevels > 1:
                monthly_pattern.columns = ['_'.join(col).strip('_') for col in monthly_pattern.columns.values]
            
            results['mes'] = monthly_pattern
        
        # Análisis por hora del día
        if 'transaction_date' in df.columns:
            df['hour'] = df['transaction_date'].dt.hour
            hourly_pattern = df.groupby('hour').agg({
                'total_spent': ['sum', 'mean', 'count']
            }).round(2)
            
            if hourly_pattern.columns.nlevels > 1:
                hourly_pattern.columns = ['_'.join(col).strip('_') for col in hourly_pattern.columns.values]
            
            results['hora'] = hourly_pattern
    
    return results


# =====================================================
# VISUALIZACIONES SIMPLES Y CLARAS
# =====================================================
def create_simple_bar_chart(data, x_col, y_col, title, color_col=None):
    """Crea gráfico de barras simple"""
    # Reset index para convertir el índice en columna
    plot_data = data.reset_index()
    
    fig = px.bar(
        plot_data,
        x=x_col,
        y=y_col,
        title=title,
        color=color_col,
        text_auto=True
    )
    fig.update_layout(
        plot_bgcolor='white',
        xaxis_title=x_col,
        yaxis_title=y_col,
        showlegend=color_col is not None
    )
    return fig


def create_box_plot(df, y_col, title):
    """Crea box plot simple"""
    try:
        fig = px.box(
            df,
            y=y_col,
            title=title,
            points="outliers"
        )
        fig.update_layout(
            plot_bgcolor='white',
            yaxis_title=y_col
        )
        return fig
    except Exception as e:
        st.error(f"Error creando box plot: {str(e)}")
        return None


def create_heatmap(df, title):
    """Crea mapa de calor de correlaciones"""
    try:
        numeric_df = df.select_dtypes(include=[np.number])
        if len(numeric_df.columns) > 1:
            corr_matrix = numeric_df.corr()
            fig = px.imshow(
                corr_matrix,
                title=title,
                color_continuous_scale='RdBu',
                text_auto='.2f',
                aspect="auto"
            )
            return fig
        return None
    except Exception as e:
        st.error(f"Error creando heatmap: {str(e)}")
        return None


# =====================================================
# INTERFAZ PRINCIPAL
# =====================================================
def main():
    # Inicializar variables de sesión si no existen
    if 'df_clean' not in st.session_state:
        st.session_state.df_clean = None
    if 'df_original' not in st.session_state:
        st.session_state.df_original = None
    if 'transformations' not in st.session_state:
        st.session_state.transformations = []
    
    # Sidebar
    with st.sidebar:
        st.title("📊 Dashboard Retail")
        st.markdown("---")
        
        # Navegación
        page = st.radio(
            "Navegación",
            ["🏠 Inicio", "🔄 Limpieza", "📈 Análisis Negocio", "📊 Visualizaciones", "📋 KPIs"]
        )
        
        # Carga de datos
        st.markdown("---")
        st.subheader("📂 Carga de Datos")
        uploaded_file = st.file_uploader("Sube tu archivo CSV", type=['csv'])
        
        if uploaded_file:
            df = load_file(uploaded_file)
            if df is not None:
                st.success(f"✅ {uploaded_file.name}")
                st.info(f"📊 {len(df)} registros, {len(df.columns)} columnas")
                
                # Procesar datos
                if st.button("🔄 Procesar Datos", type="primary"):
                    with st.spinner("Limpiando y procesando datos..."):
                        df_clean, df_original, transformations = clean_retail_data(df)
                        st.session_state.df = df
                        st.session_state.df_clean = df_clean
                        st.session_state.df_original = df_original
                        st.session_state.transformations = transformations
                        st.success("✅ Datos procesados")
        else:
            st.info("👆 Sube un archivo CSV para comenzar")
    
    # Páginas principales
    if page == "🏠 Inicio":
        show_home_page()
    elif page == "🔄 Limpieza":
        show_cleaning_page()
    elif page == "📈 Análisis Negocio":
        show_business_analysis_page()
    elif page == "📊 Visualizaciones":
        show_visualizations_page()
    elif page == "📋 KPIs":
        show_kpis_page()


# =====================================================
# PÁGINA: INICIO
# =====================================================
def show_home_page():
    st.markdown('<h1 class="main-header">🏠 Dashboard Retail Inteligente</h1>', unsafe_allow_html=True)
    
    if st.session_state.df_clean is not None:
        df = st.session_state.df_clean
        
        # Métricas principales
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            if 'total_spent' in df.columns:
                total_sales = df['total_spent'].sum()
                st.metric("💰 Ventas Totales", f"${total_sales:,.0f}")
        
        with col2:
            if 'total_spent' in df.columns:
                avg_ticket = df['total_spent'].mean()
                st.metric("🎫 Ticket Promedio", f"${avg_ticket:,.2f}")
        
        with col3:
            st.metric("📊 Transacciones", f"{len(df):,}")
        
        with col4:
            if 'category' in df.columns:
                unique_categories = df['category'].nunique()
                st.metric("🏷️ Categorías", f"{unique_categories}")
        
        # Resumen del dataset
        st.markdown("---")
        st.markdown('<h2 class="sub-header">📋 Resumen del Dataset</h2>', unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Primeros registros:**")
            st.dataframe(df.head(), use_container_width=True)
        
        with col2:
            st.write("**Información de columnas:**")
            info_df = pd.DataFrame({
                'Columna': df.columns,
                'Tipo': df.dtypes.values,
                'No Nulos': df.notnull().sum().values
            })
            st.dataframe(info_df, use_container_width=True)
        
        # Descripción estadística
        st.markdown("---")
        st.markdown('<h2 class="sub-header">📊 Estadísticas Descriptivas</h2>', unsafe_allow_html=True)
        
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 0:
            st.dataframe(df[numeric_cols].describe(), use_container_width=True)
    else:
        st.info("👈 Sube un archivo CSV y procésalo para ver los datos")


# =====================================================
# PÁGINA: LIMPIEZA
# =====================================================
def show_cleaning_page():
    st.markdown('<h1 class="main-header">🔄 Limpieza de Datos</h1>', unsafe_allow_html=True)
    
    if st.session_state.df_clean is not None and st.session_state.transformations:
        df = st.session_state.df_clean
        transformations = st.session_state.transformations
        df_original = st.session_state.df_original if st.session_state.df_original is not None else df
        
        # Resumen de transformaciones
        st.markdown('<h3 class="sub-header">📋 Transformaciones Aplicadas</h3>', unsafe_allow_html=True)
        
        with st.expander("Ver detalles de limpieza", expanded=True):
            for transform in transformations:
                st.write(transform)
        
        # Comparación
        st.markdown("---")
        st.markdown('<h3 class="sub-header">👁️ Comparación Antes/Después</h3>', unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Datos Originales (muestra)**")
            st.dataframe(df_original.head(), use_container_width=True)
            st.write(f"Registros: {len(df_original):,}")
            st.write(f"Valores nulos: {df_original.isnull().sum().sum():,}")
        
        with col2:
            st.write("**Datos Limpios (muestra)**")
            st.dataframe(df.head(), use_container_width=True)
            st.write(f"Registros: {len(df):,}")
            st.write(f"Valores nulos: {df.isnull().sum().sum():,}")
        
        # Descarga de datos limpios
        st.markdown("---")
        st.markdown('<h3 class="sub-header">💾 Exportar Datos</h3>', unsafe_allow_html=True)
        
        csv = df.to_csv(index=False)
        st.download_button(
            "📥 Descargar CSV Limpio",
            csv,
            "datos_retail_limpios.csv",
            "text/csv",
            key='download-csv'
        )
    else:
        st.warning("⚠️ No hay datos procesados. Por favor, sube un archivo y procésalo en la página de Inicio.")


# =====================================================
# PÁGINA: ANÁLISIS DE NEGOCIO
# =====================================================
def show_business_analysis_page():
    st.markdown('<h1 class="main-header">📈 Análisis de Negocio</h1>', unsafe_allow_html=True)
    
    if st.session_state.df_clean is not None:
        df = st.session_state.df_clean
        
        # Pregunta 1: Rentabilidad por categoría
        st.markdown('<h3 class="sub-header">1️⃣ ¿Qué categorías generan mayor ingreso y cuáles menor rentabilidad?</h3>', unsafe_allow_html=True)
        
        category_analysis = analyze_category_profitability(df)
        
        if category_analysis is not None:
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("**Top 5 Categorías por Ingreso:**")
                top_categories = category_analysis.head(5)
                st.dataframe(top_categories, use_container_width=True)
                
                # Gráfico de barras
                fig = create_simple_bar_chart(
                    top_categories,
                    'index',
                    'ingreso_total',
                    'Top 5 Categorías por Ingreso Total',
                    color_col='index'
                )
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                st.write("**Bottom 5 Categorías por Rentabilidad:**")
                bottom_rentability = category_analysis.sort_values('rentabilidad').head(5)
                st.dataframe(bottom_rentability, use_container_width=True)
                
                fig = create_simple_bar_chart(
                    bottom_rentability,
                    'index',
                    'rentabilidad',
                    'Bottom 5 Categorías por Rentabilidad',
                    color_col='index'
                )
                st.plotly_chart(fig, use_container_width=True)
            
            # Insights
            if not category_analysis.empty:
                top_category = category_analysis.index[0]
                top_income = category_analysis.iloc[0]['ingreso_total']
                bottom_category = category_analysis.sort_values('rentabilidad').index[0]
                bottom_rent = category_analysis.sort_values('rentabilidad').iloc[0]['rentabilidad']
                
                st.info(f"""
                **💡 Insights:**
                - Categoría con mayor ingreso: **{top_category}** (${top_income:,.0f})
                - Categoría con menor rentabilidad: **{bottom_category}** (${bottom_rent:,.2f} por transacción)
                """)
        
        # Pregunta 2: Segmentos de clientes
        st.markdown("---")
        st.markdown('<h3 class="sub-header">2️⃣ ¿Qué segmentos de clientes tienen el ticket promedio más alto?</h3>', unsafe_allow_html=True)
        
        segment_analysis = analyze_customer_segments(df)
        
        if segment_analysis:
            tabs = st.tabs(list(segment_analysis.keys()))
            
            for i, (segment_type, analysis) in enumerate(segment_analysis.items()):
                with tabs[i]:
                    st.write(f"**Análisis por {segment_type}:**")
                    st.dataframe(analysis, use_container_width=True)
                    
                    # Gráfico para el top 5
                    if 'total_spent_mean' in analysis.columns:
                        top_segments = analysis.head(5)
                        fig = create_simple_bar_chart(
                            top_segments,
                            'index',
                            'total_spent_mean',
                            f'Top 5 {segment_type.title()} por Ticket Promedio',
                            color_col='index'
                        )
                        st.plotly_chart(fig, use_container_width=True)
        
        # Pregunta 3: Patrones temporales
        st.markdown("---")
        st.markdown('<h3 class="sub-header">3️⃣ ¿Existen patrones temporales en las ventas?</h3>', unsafe_allow_html=True)
        
        temporal_analysis = analyze_temporal_patterns(df)
        
        if temporal_analysis:
            tabs = st.tabs(list(temporal_analysis.keys()))
            
            for i, (pattern_type, analysis) in enumerate(temporal_analysis.items()):
                with tabs[i]:
                    st.write(f"**Patrones por {pattern_type}:**")
                    st.dataframe(analysis, use_container_width=True)
                    
                    # Gráfico
                    if 'total_spent_sum' in analysis.columns:
                        fig = create_simple_bar_chart(
                            analysis,
                            'index',
                            'total_spent_sum',
                            f'Ventas Totales por {pattern_type.title()}',
                            color_col='index'
                        )
                        st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("⚠️ No hay datos procesados. Por favor, sube un archivo y procésalo en la página de Inicio.")


# =====================================================
# PÁGINA: VISUALIZACIONES
# =====================================================
def show_visualizations_page():
    st.markdown('<h1 class="main-header">📊 Visualizaciones</h1>', unsafe_allow_html=True)
    
    if st.session_state.df_clean is not None:
        df = st.session_state.df_clean
        
        # Seleccionar tipo de visualización
        viz_type = st.selectbox(
            "Seleccionar tipo de visualización",
            ["Distribuciones", "Comparaciones", "Relaciones", "Temporal"]
        )
        
        if viz_type == "Distribuciones":
            st.markdown('<h3 class="sub-header">📈 Distribuciones</h3>', unsafe_allow_html=True)
            
            col1, col2 = st.columns(2)
            
            with col1:
                if 'total_spent' in df.columns:
                    fig = create_box_plot(df, 'total_spent', 'Distribución de Montos de Venta')
                    if fig:
                        st.plotly_chart(fig, use_container_width=True)
                    else:
                        st.warning("No se pudo crear el gráfico de caja para total_spent")
            
            with col2:
                if 'quantity' in df.columns:
                    fig = create_box_plot(df, 'quantity', 'Distribución de Cantidades Vendidas')
                    if fig:
                        st.plotly_chart(fig, use_container_width=True)
                    else:
                        st.warning("No se pudo crear el gráfico de caja para quantity")
            
            # Histograma simple
            if 'total_spent' in df.columns:
                try:
                    fig = px.histogram(
                        df,
                        x='total_spent',
                        nbins=30,
                        title='Distribución de Montos de Venta',
                        labels={'total_spent': 'Monto ($)'}
                    )
                    st.plotly_chart(fig, use_container_width=True)
                except Exception as e:
                    st.error(f"Error creando histograma: {str(e)}")
        
        elif viz_type == "Comparaciones":
            st.markdown('<h3 class="sub-header">🔗 Comparaciones</h3>', unsafe_allow_html=True)
            
            # Comparación por categoría
            if 'category' in df.columns and 'total_spent' in df.columns:
                try:
                    category_sales = df.groupby('category')['total_spent'].sum().sort_values(ascending=False).head(10)
                    
                    fig = px.bar(
                        category_sales.reset_index(),
                        x='total_spent',
                        y='category',
                        orientation='h',
                        title='Top 10 Categorías por Ventas',
                        labels={'total_spent': 'Ventas Totales', 'category': 'Categoría'}
                    )
                    st.plotly_chart(fig, use_container_width=True)
                except Exception as e:
                    st.error(f"Error creando gráfico de categorías: {str(e)}")
            
            # Comparación por ubicación
            if 'location' in df.columns and 'total_spent' in df.columns:
                try:
                    location_sales = df.groupby('location')['total_spent'].sum().sort_values(ascending=False)
                    
                    fig = px.bar(
                        location_sales.reset_index(),
                        x='location',
                        y='total_spent',
                        title='Ventas por Ubicación',
                        labels={'location': 'Ubicación', 'total_spent': 'Ventas Totales'}
                    )
                    st.plotly_chart(fig, use_container_width=True)
                except Exception as e:
                    st.error(f"Error creando gráfico de ubicaciones: {str(e)}")
        
        elif viz_type == "Relaciones":
            st.markdown('<h3 class="sub-header">🔗 Relaciones entre Variables</h3>', unsafe_allow_html=True)
            
            # Mapa de calor de correlaciones
            heatmap = create_heatmap(df, 'Correlaciones entre Variables Numéricas')
            if heatmap:
                st.plotly_chart(heatmap, use_container_width=True)
            else:
                st.info("No hay suficientes variables numéricas para crear un mapa de calor")
            
            # Scatter plot simple
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            if len(numeric_cols) >= 2:
                col1, col2 = st.columns(2)
                
                with col1:
                    x_var = st.selectbox("Variable X", numeric_cols, index=0)
                
                with col2:
                    y_var = st.selectbox("Variable Y", numeric_cols, index=1 if len(numeric_cols) > 1 else 0)
                
                try:
                    fig = px.scatter(
                        df,
                        x=x_var,
                        y=y_var,
                        title=f'{x_var} vs {y_var}',
                        trendline='ols'
                    )
                    st.plotly_chart(fig, use_container_width=True)
                except Exception as e:
                    st.error(f"Error creando scatter plot: {str(e)}")
        
        elif viz_type == "Temporal":
            st.markdown('<h3 class="sub-header">📅 Análisis Temporal</h3>', unsafe_allow_html=True)
            
            if 'transaction_date' in df.columns and 'total_spent' in df.columns:
                try:
                    # Serie temporal diaria
                    df_temp = df.copy()
                    df_temp['date'] = df_temp['transaction_date'].dt.date
                    daily_sales = df_temp.groupby('date')['total_spent'].sum().reset_index()
                    
                    fig = px.line(
                        daily_sales,
                        x='date',
                        y='total_spent',
                        title='Ventas Diarias',
                        labels={'date': 'Fecha', 'total_spent': 'Ventas Totales'}
                    )
                    st.plotly_chart(fig, use_container_width=True)
                except Exception as e:
                    st.error(f"Error creando gráfico temporal diario: {str(e)}")
                
                # Ventas por día de semana
                if 'weekday' in df.columns:
                    try:
                        weekday_sales = df.groupby('weekday')['total_spent'].sum().reset_index()
                        day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
                        weekday_sales['weekday'] = pd.Categorical(weekday_sales['weekday'], categories=day_order, ordered=True)
                        weekday_sales = weekday_sales.sort_values('weekday')
                        
                        fig = px.bar(
                            weekday_sales,
                            x='weekday',
                            y='total_spent',
                            title='Ventas por Día de la Semana',
                            labels={'weekday': 'Día', 'total_spent': 'Ventas Totales'}
                        )
                        st.plotly_chart(fig, use_container_width=True)
                    except Exception as e:
                        st.error(f"Error creando gráfico por día de semana: {str(e)}")
    else:
        st.warning("⚠️ No hay datos procesados. Por favor, sube un archivo y procésalo en la página de Inicio.")


# =====================================================
# PÁGINA: KPIs
# =====================================================
def show_kpis_page():
    st.markdown('<h1 class="main-header">📋 Panel de KPIs</h1>', unsafe_allow_html=True)
    
    if st.session_state.df_clean is not None:
        df = st.session_state.df_clean
        
        # KPIs principales
        st.markdown('<h3 class="sub-header">📊 KPIs Principales</h3>', unsafe_allow_html=True)
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            if 'total_spent' in df.columns:
                total_sales = df['total_spent'].sum()
                st.metric("💰 Ventas Totales", f"${total_sales:,.0f}")
        
        with col2:
            if 'total_spent' in df.columns:
                avg_ticket = df['total_spent'].mean()
                st.metric("🎫 Ticket Promedio", f"${avg_ticket:,.2f}")
        
        with col3:
            transactions = len(df)
            st.metric("🛒 Transacciones", f"{transactions:,}")
        
        with col4:
            if 'quantity' in df.columns:
                total_units = df['quantity'].sum()
                st.metric("📦 Unidades Vendidas", f"{total_units:,}")
        
        # KPIs por categoría
        st.markdown("---")
        st.markdown('<h3 class="sub-header">🏷️ KPIs por Categoría</h3>', unsafe_allow_html=True)
        
        if 'category' in df.columns:
            category_kpis = df.groupby('category').agg({
                'total_spent': ['sum', 'mean', 'count']
            }).round(2)
            
            if category_kpis.columns.nlevels > 1:
                category_kpis.columns = ['_'.join(col).strip('_') for col in category_kpis.columns.values]
            
            st.dataframe(category_kpis, use_container_width=True)
        
        # KPIs temporales
        st.markdown("---")
        st.markdown('<h3 class="sub-header">📅 KPIs Temporales</h3>', unsafe_allow_html=True)
        
        if 'transaction_date' in df.columns:
            # Últimos 30 días vs período anterior
            df_temp = df.copy()
            df_temp['date'] = df_temp['transaction_date'].dt.date
            
            if len(df_temp) > 0:
                latest_date = df_temp['date'].max()
                
                last_30_days = latest_date - pd.Timedelta(days=30)
                previous_30_days = last_30_days - pd.Timedelta(days=30)
                
                sales_last_30 = df_temp[df_temp['date'] >= last_30_days]['total_spent'].sum()
                sales_previous_30 = df_temp[(df_temp['date'] >= previous_30_days) & 
                                           (df_temp['date'] < last_30_days)]['total_spent'].sum()
                
                growth = ((sales_last_30 - sales_previous_30) / sales_previous_30 * 100) if sales_previous_30 > 0 else 0
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.metric("💰 Ventas últimos 30 días", f"${sales_last_30:,.0f}")
                
                with col2:
                    st.metric("📈 Crecimiento vs período anterior", f"{growth:.1f}%")
    else:
        st.warning("⚠️ No hay datos procesados. Por favor, sube un archivo y procésalo en la página de Inicio.")


# =====================================================
# EJECUCIÓN PRINCIPAL
# =====================================================
if __name__ == "__main__":
    main()