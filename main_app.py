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
import requests
import json

warnings.filterwarnings('ignore')

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
    try:
        df = pd.read_csv(file)
        return df
    except Exception as e:
        st.error(f"Error al cargar archivo: {str(e)}")
        return None


# =====================================================
# LIMPIEZA DE DATOS (basada en el notebook)
# =====================================================
def clean_retail_data(df):
    transformations = []
    df_original = df.copy()
    initial_rows = len(df)
    initial_cols = len(df.columns)
    transformations.append(f"📊 **INICIO:** {initial_rows:,} registros, {initial_cols} columnas")

    # Eliminar columnas innecesarias
    cols_to_drop = []
    if 'Transaction ID' in df.columns:
        cols_to_drop.append('Transaction ID')
    if 'Item' in df.columns:
        cols_to_drop.append('Item')
    if cols_to_drop:
        df = df.drop(columns=cols_to_drop)
        transformations.append(f"🗑️ Eliminadas: {', '.join(cols_to_drop)}")

    # Normalizar nombres a snake_case
    df.columns = df.columns.str.lower().str.replace(' ', '_', regex=True)
    transformations.append("🔄 Nombres en snake_case")

    # Manejo de valores faltantes
    transformations.append("\n🔍 **VALORES FALTANTES:**")
    if 'discount_applied' in df.columns:
        miss = df['discount_applied'].isnull().sum()
        df['discount_applied'] = df['discount_applied'].fillna(0).astype(int)
        transformations.append(f"   • discount_applied: {miss} nulos → 0")

    missing_before = df.isnull().sum().sum()
    df = df.dropna()
    missing_after = df.isnull().sum().sum()
    transformations.append(f"   • Filas eliminadas por nulos: {missing_before - missing_after}")

    # Conversión de tipos
    transformations.append("\n🔄 **TIPOS DE DATOS:**")
    cat_cols = ['category', 'payment_method', 'location']
    for col in cat_cols:
        if col in df.columns:
            df[col] = df[col].astype('category')
            transformations.append(f"   • {col} → category")

    if 'transaction_date' in df.columns:
        df['transaction_date'] = pd.to_datetime(df['transaction_date'], errors='coerce')
        inv = df['transaction_date'].isnull().sum()
        if inv > 0:
            df = df.dropna(subset=['transaction_date'])
            transformations.append(f"   • transaction_date: {inv} fechas inválidas eliminadas")

    num_cols = ['quantity', 'price_per_unit', 'total_spent']
    for col in num_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    # Feature engineering temporal
    transformations.append("\n⚙️ **FEATURE ENGINEERING:**")
    if 'transaction_date' in df.columns:
        df['year'] = df['transaction_date'].dt.year
        df['month'] = df['transaction_date'].dt.month
        df['day'] = df['transaction_date'].dt.day
        df['weekday'] = df['transaction_date'].dt.day_name()
        df['month_name'] = df['transaction_date'].dt.month_name()
        transformations.append("   • Año, mes, día, día_semana, mes_nombre")

    final_rows = len(df)
    transformations.append(f"\n✅ **FINAL:** {final_rows:,} registros ({initial_rows-final_rows} eliminados, {((initial_rows-final_rows)/initial_rows*100):.1f}%)")
    return df, df_original, transformations


# =====================================================
# ANÁLISIS DE NEGOCIO (preguntas 1,2,3)
# =====================================================
def analyze_category_profitability(df):
    """Pregunta 1: Ingreso y rentabilidad por categoría"""
    if 'category' not in df.columns or 'total_spent' not in df.columns:
        return None
    agg = df.groupby('category').agg(
        ingreso_total=('total_spent', 'sum'),
        ticket_promedio=('total_spent', 'mean'),
        transacciones=('total_spent', 'count')
    ).round(2)
    agg['rentabilidad'] = (agg['ingreso_total'] / agg['transacciones']).round(2)
    return agg.sort_values('ingreso_total', ascending=False)


def analyze_customer_segments(df):
    """
    Pregunta 2: Segmentos de clientes con mayor ticket promedio.
    Retorna dict con DataFrames: ubicación, método_pago, categoría.
    """
    results = {}
    if 'location' in df.columns and 'total_spent' in df.columns:
        loc = df.groupby('location')['total_spent'].mean().round(2).reset_index()
        loc.columns = ['ubicacion', 'ticket_promedio']
        results['ubicacion'] = loc.sort_values('ticket_promedio', ascending=False)

    if 'payment_method' in df.columns and 'total_spent' in df.columns:
        pay = df.groupby('payment_method')['total_spent'].mean().round(2).reset_index()
        pay.columns = ['metodo_pago', 'ticket_promedio']
        results['metodo_pago'] = pay.sort_values('ticket_promedio', ascending=False)

    if 'category' in df.columns and 'total_spent' in df.columns:
        cat = df.groupby('category')['total_spent'].mean().round(2).reset_index()
        cat.columns = ['categoria', 'ticket_promedio']
        results['categoria'] = cat.sort_values('ticket_promedio', ascending=False)

    return results


def analyze_temporal_patterns(df):
    """Pregunta 3: Patrones temporales (día, mes, hora)"""
    results = {}
    if 'transaction_date' not in df.columns or 'total_spent' not in df.columns:
        return results

    # Día de semana
    if 'weekday' in df.columns:
        dia = df.groupby('weekday')['total_spent'].sum().round(2).reset_index()
        dia.columns = ['dia_semana', 'ventas_totales']
        orden = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        dia['dia_semana'] = pd.Categorical(dia['dia_semana'], categories=orden, ordered=True)
        results['dia_semana'] = dia.sort_values('dia_semana')

    # Mes
    if 'month_name' in df.columns:
        mes = df.groupby('month_name')['total_spent'].sum().round(2).reset_index()
        mes.columns = ['mes', 'ventas_totales']
        orden_meses = ['January', 'February', 'March', 'April', 'May', 'June',
                       'July', 'August', 'September', 'October', 'November', 'December']
        mes['mes'] = pd.Categorical(mes['mes'], categories=orden_meses, ordered=True)
        results['mes'] = mes.sort_values('mes')

    # Hora (copia local para no modificar df original)
    df_hour = df.copy()
    df_hour['hour'] = df_hour['transaction_date'].dt.hour
    hora = df_hour.groupby('hour')['total_spent'].sum().round(2).reset_index()
    hora.columns = ['hora', 'ventas_totales']
    results['hora'] = hora.sort_values('hora')

    return results


# =====================================================
# VISUALIZACIONES (funciones auxiliares)
# =====================================================
def create_simple_bar_chart(df, x_col, y_col, title, color_col=None):
    """Gráfico de barras simple con Plotly"""
    fig = px.bar(
        df,
        x=x_col,
        y=y_col,
        color=color_col,
        title=title,
        text_auto=True
    )
    fig.update_layout(
        template="plotly_white",
        xaxis_title=x_col.replace('_', ' ').title(),
        yaxis_title=y_col.replace('_', ' ').title()
    )
    return fig


def create_box_plot(df, y_col, title):
    try:
        fig = px.box(df, y=y_col, title=title, points="outliers")
        fig.update_layout(plot_bgcolor='white', yaxis_title=y_col)
        return fig
    except Exception as e:
        st.error(f"Error box plot: {e}")
        return None


def create_heatmap(df, title):
    try:
        numeric_df = df.select_dtypes(include=[np.number])
        if len(numeric_df.columns) > 1:
            corr = numeric_df.corr()
            fig = px.imshow(corr, title=title, color_continuous_scale='RdBu',
                            text_auto='.2f', aspect="auto")
            return fig
        return None
    except Exception as e:
        st.error(f"Error heatmap: {e}")
        return None


# =====================================================
# INTERFAZ PRINCIPAL
# =====================================================
def main():
    # Estado de sesión
    for var in ['df_clean', 'df_original', 'transformations', 'groq_api_key', 'ai_insights']:
        if var not in st.session_state:
            st.session_state[var] = None if var != 'transformations' else []

    # Sidebar
    with st.sidebar:
        st.title("📊 Dashboard Retail")
        st.markdown("---")

        page = st.radio(
            "Navegación",
            ["🏠 Inicio", "🔄 Limpieza", "📈 Análisis Negocio",
             "📊 Visualizaciones", "📋 KPIs", "🤖 Insights IA"]
        )

        st.markdown("---")
        st.subheader("📂 Carga de Datos")
        uploaded_file = st.file_uploader("Sube tu archivo CSV", type=['csv'])

        if uploaded_file:
            df = load_file(uploaded_file)
            if df is not None:
                st.success(f"✅ {uploaded_file.name}")
                st.info(f"📊 {len(df)} registros, {len(df.columns)} columnas")

                if st.button("🔄 Procesar Datos", type="primary"):
                    with st.spinner("Limpiando y procesando..."):
                        df_clean, df_orig, trans = clean_retail_data(df)
                        st.session_state.df_clean = df_clean
                        st.session_state.df_original = df_orig
                        st.session_state.transformations = trans
                        st.success("✅ Datos procesados")
        else:
            st.info("👆 Sube un archivo CSV para comenzar")

        st.markdown("---")
        st.subheader("🤖 Configuración IA")
        api_key = st.text_input("Groq API Key", type="password",
                                help="Obtén tu API Key en console.groq.com")
        if api_key:
            st.session_state.groq_api_key = api_key
            st.success("✅ API Key configurada")
        else:
            st.session_state.groq_api_key = None

    # Navegación de páginas
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
    elif page == "🤖 Insights IA":
        show_ai_insights_page()


# =====================================================
# PÁGINA: INICIO
# =====================================================
def show_home_page():
    st.markdown('<h1 class="main-header">🏠 Dashboard Retail Inteligente</h1>', unsafe_allow_html=True)
    if st.session_state.df_clean is None:
        st.info("👈 Sube y procesa un archivo CSV para comenzar")
        return

    df = st.session_state.df_clean
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        if 'total_spent' in df.columns:
            st.metric("💰 Ventas Totales", f"${df['total_spent'].sum():,.0f}")
    with col2:
        if 'total_spent' in df.columns:
            st.metric("🎫 Ticket Promedio", f"${df['total_spent'].mean():,.2f}")
    with col3:
        st.metric("📊 Transacciones", f"{len(df):,}")
    with col4:
        if 'category' in df.columns:
            st.metric("🏷️ Categorías", f"{df['category'].nunique()}")

    st.markdown("---")
    st.markdown('<h2 class="sub-header">📋 Resumen del Dataset</h2>', unsafe_allow_html=True)
    col1, col2 = st.columns(2)
    with col1:
        st.write("**Primeros registros**")
        st.dataframe(df.head(), use_container_width=True)
    with col2:
        st.write("**Información de columnas**")
        info = pd.DataFrame({
            'Columna': df.columns,
            'Tipo': df.dtypes.values,
            'No Nulos': df.notnull().sum().values
        })
        st.dataframe(info, use_container_width=True)

    st.markdown("---")
    st.markdown('<h2 class="sub-header">📊 Estadísticas Descriptivas</h2>', unsafe_allow_html=True)
    numeric = df.select_dtypes(include=[np.number])
    if not numeric.empty:
        st.dataframe(numeric.describe(), use_container_width=True)


# =====================================================
# PÁGINA: LIMPIEZA
# =====================================================
def show_cleaning_page():
    st.markdown('<h1 class="main-header">🔄 Limpieza de Datos</h1>', unsafe_allow_html=True)
    if st.session_state.df_clean is None:
        st.warning("⚠️ No hay datos procesados. Ve a Inicio y procesa un archivo.")
        return

    df = st.session_state.df_clean
    trans = st.session_state.transformations
    df_orig = st.session_state.df_original if st.session_state.df_original is not None else df

    st.markdown('<h3 class="sub-header">📋 Transformaciones Aplicadas</h3>', unsafe_allow_html=True)
    with st.expander("Ver detalles", expanded=True):
        for t in trans:
            st.write(t)

    st.markdown("---")
    st.markdown('<h3 class="sub-header">👁️ Comparación Antes/Después</h3>', unsafe_allow_html=True)
    col1, col2 = st.columns(2)
    with col1:
        st.write("**Datos Originales (muestra)**")
        st.dataframe(df_orig.head(), use_container_width=True)
        st.write(f"Registros: {len(df_orig):,}  |  Nulos: {df_orig.isnull().sum().sum():,}")
    with col2:
        st.write("**Datos Limpios (muestra)**")
        st.dataframe(df.head(), use_container_width=True)
        st.write(f"Registros: {len(df):,}  |  Nulos: {df.isnull().sum().sum():,}")

    st.markdown("---")
    st.markdown('<h3 class="sub-header">💾 Exportar Datos</h3>', unsafe_allow_html=True)
    csv = df.to_csv(index=False)
    st.download_button("📥 Descargar CSV Limpio", csv, "datos_retail_limpios.csv", "text/csv")


# =====================================================
# PÁGINA: ANÁLISIS DE NEGOCIO (preguntas 1,2,3)
# =====================================================
def show_business_analysis_page():
    st.markdown('<h1 class="main-header">📈 Análisis de Negocio</h1>', unsafe_allow_html=True)
    if st.session_state.df_clean is None:
        st.warning("⚠️ No hay datos procesados. Ve a Inicio y procesa un archivo.")
        return

    df = st.session_state.df_clean

    # ---------- Pregunta 1: Rentabilidad por categoría ----------
    st.markdown('<h3 class="sub-header">1️⃣ ¿Qué categorías generan mayor ingreso y cuáles menor rentabilidad?</h3>', unsafe_allow_html=True)
    cat_analysis = analyze_category_profitability(df)
    if cat_analysis is not None:
        col1, col2 = st.columns(2)
        with col1:
            st.write("**Top 5 por Ingreso**")
            top = cat_analysis.head(5).reset_index()
            st.dataframe(top, use_container_width=True)
            fig = create_simple_bar_chart(top, 'category', 'ingreso_total',
                                          'Top 5 Categorías por Ingreso', color_col='category')
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            st.write("**Bottom 5 por Rentabilidad**")
            bottom = cat_analysis.sort_values('rentabilidad').head(5).reset_index()
            st.dataframe(bottom, use_container_width=True)
            fig = create_simple_bar_chart(bottom, 'category', 'rentabilidad',
                                          'Bottom 5 Categorías por Rentabilidad', color_col='category')
            st.plotly_chart(fig, use_container_width=True)

        st.info(f"""
        **💡 Insights:**
        - Mayor ingreso: **{cat_analysis.index[0]}** (${cat_analysis.iloc[0]['ingreso_total']:,.0f})
        - Menor rentabilidad: **{cat_analysis.sort_values('rentabilidad').index[0]}** (${cat_analysis.sort_values('rentabilidad').iloc[0]['rentabilidad']:,.2f}/transacción)
        """)

    # ---------- Pregunta 2: Segmentos de clientes ----------
    st.markdown("---")
    st.markdown('<h3 class="sub-header">2️⃣ ¿Qué segmentos de clientes tienen el ticket promedio más alto?</h3>', unsafe_allow_html=True)
    segment_data = analyze_customer_segments(df)
    if segment_data:
        tabs = st.tabs(list(segment_data.keys()))
        for tab, (key, df_seg) in zip(tabs, segment_data.items()):
            with tab:
                st.write(f"**Análisis por {key.replace('_', ' ').title()}**")
                st.dataframe(df_seg, use_container_width=True)
                if not df_seg.empty:
                    x_col = df_seg.columns[0]
                    fig = create_simple_bar_chart(df_seg.head(5), x_col, 'ticket_promedio',
                                                  f'Top 5 {key.title()} por Ticket Promedio',
                                                  color_col=x_col)
                    st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No hay suficientes datos para analizar segmentos.")

    # ---------- Pregunta 3: Patrones temporales ----------
    st.markdown("---")
    st.markdown('<h3 class="sub-header">3️⃣ ¿Existen patrones temporales en las ventas?</h3>', unsafe_allow_html=True)
    temporal = analyze_temporal_patterns(df)
    if temporal:
        tabs = st.tabs(list(temporal.keys()))
        for tab, (key, df_temp) in zip(tabs, temporal.items()):
            with tab:
                st.write(f"**Patrón por {key.replace('_', ' ').title()}**")
                st.dataframe(df_temp, use_container_width=True)
                x_col = df_temp.columns[0]
                fig = create_simple_bar_chart(df_temp, x_col, 'ventas_totales',
                                              f'Ventas Totales por {x_col.title()}',
                                              color_col=x_col)
                st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No hay datos temporales suficientes.")


# =====================================================
# PÁGINA: VISUALIZACIONES (EDA)
# =====================================================
def show_visualizations_page():
    st.markdown('<h1 class="main-header">📊 Análisis Exploratorio (EDA)</h1>', unsafe_allow_html=True)
    if st.session_state.df_clean is None:
        st.warning("⚠️ No hay datos procesados.")
        return

    df = st.session_state.df_clean
    tab_uni, tab_bi, tab_temp, tab_report = st.tabs(
        ["📈 Univariado", "🔗 Bivariado", "📅 Temporal", "🧾 Reporte"]
    )

    with tab_uni:
        st.subheader("Distribución de Variables Clave")
        col1, col2 = st.columns(2)
        with col1:
            if 'total_spent' in df.columns:
                fig = px.histogram(df, x='total_spent', nbins=30, title='Distribución Monto de Venta')
                st.plotly_chart(fig, use_container_width=True)
        with col2:
            if 'quantity' in df.columns:
                fig = px.histogram(df, x='quantity', nbins=20, title='Distribución Cantidad')
                st.plotly_chart(fig, use_container_width=True)
        if 'total_spent' in df.columns:
            fig = create_box_plot(df, 'total_spent', 'Outliers en Montos')
            if fig:
                st.plotly_chart(fig, use_container_width=True)

    with tab_bi:
        st.subheader("Relaciones y Comparaciones")
        if {'category', 'total_spent'}.issubset(df.columns):
            cat_sales = df.groupby('category')['total_spent'].sum().reset_index()
            cat_sales = cat_sales.sort_values('total_spent', ascending=False).head(10)
            fig = px.bar(cat_sales, x='total_spent', y='category', orientation='h',
                         title='Top 10 Categorías por Ventas')
            st.plotly_chart(fig, use_container_width=True)

        if {'location', 'total_spent'}.issubset(df.columns):
            loc_sales = df.groupby('location')['total_spent'].sum().reset_index()
            loc_sales = loc_sales.sort_values('total_spent', ascending=False)
            fig = px.bar(loc_sales, x='location', y='total_spent', title='Ventas por Ubicación')
            st.plotly_chart(fig, use_container_width=True)

        numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
        if len(numeric_cols) >= 2:
            col1, col2 = st.columns(2)
            x_var = col1.selectbox("Variable X", numeric_cols)
            y_var = col2.selectbox("Variable Y", numeric_cols, index=1)
            fig = px.scatter(df, x=x_var, y=y_var, trendline='ols', title=f'{x_var} vs {y_var}')
            st.plotly_chart(fig, use_container_width=True)

        heat = create_heatmap(df, 'Correlaciones')
        if heat:
            st.plotly_chart(heat, use_container_width=True)

    with tab_temp:
        st.subheader("Patrones Temporales")
        if {'transaction_date', 'total_spent'}.issubset(df.columns):
            df_d = df.copy()
            df_d['date'] = df_d['transaction_date'].dt.date
            daily = df_d.groupby('date')['total_spent'].sum().reset_index()
            fig = px.line(daily, x='date', y='total_spent', title='Ventas Diarias')
            st.plotly_chart(fig, use_container_width=True)

        if 'weekday' in df.columns and 'total_spent' in df.columns:
            wd = df.groupby('weekday')['total_spent'].sum().reset_index()
            orden = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
            wd['weekday'] = pd.Categorical(wd['weekday'], categories=orden, ordered=True)
            wd = wd.sort_values('weekday')
            fig = px.bar(wd, x='weekday', y='total_spent', title='Ventas por Día de Semana')
            st.plotly_chart(fig, use_container_width=True)

    with tab_report:
        st.subheader("Resumen Ejecutivo")
        st.metric("Ventas Totales", f"${df['total_spent'].sum():,.0f}")
        st.metric("Ticket Promedio", f"${df['total_spent'].mean():,.2f}")
        st.metric("Transacciones", len(df))
        with st.expander("Ver estadísticas descriptivas"):
            st.dataframe(df.describe())
        with st.expander("Descargar datos limpios"):
            st.download_button("📥 Descargar CSV", df.to_csv(index=False), "datos_limpios.csv", "text/csv")


# =====================================================
# PÁGINA: KPIs
# =====================================================
def show_kpis_page():
    st.markdown('<h1 class="main-header">📋 Panel de KPIs</h1>', unsafe_allow_html=True)
    if st.session_state.df_clean is None:
        st.warning("⚠️ No hay datos procesados.")
        return

    df = st.session_state.df_clean
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        if 'total_spent' in df.columns:
            st.metric("💰 Ventas Totales", f"${df['total_spent'].sum():,.0f}")
    with col2:
        if 'total_spent' in df.columns:
            st.metric("🎫 Ticket Promedio", f"${df['total_spent'].mean():,.2f}")
    with col3:
        st.metric("🛒 Transacciones", f"{len(df):,}")
    with col4:
        if 'quantity' in df.columns:
            st.metric("📦 Unidades", f"{df['quantity'].sum():,.0f}")

    if 'category' in df.columns:
        st.markdown("---")
        st.markdown('<h3 class="sub-header">🏷️ KPIs por Categoría</h3>', unsafe_allow_html=True)
        kpi_cat = df.groupby('category').agg(
            ventas=('total_spent', 'sum'),
            ticket=('total_spent', 'mean'),
            transacciones=('total_spent', 'count')
        ).round(2).sort_values('ventas', ascending=False)
        st.dataframe(kpi_cat, use_container_width=True)

    if 'transaction_date' in df.columns:
        st.markdown("---")
        st.markdown('<h3 class="sub-header">📅 KPIs Temporales</h3>', unsafe_allow_html=True)
        df_temp = df.copy()
        df_temp['date'] = df_temp['transaction_date'].dt.date
        if not df_temp.empty:
            latest = df_temp['date'].max()
            last30 = latest - pd.Timedelta(days=30)
            prev30 = last30 - pd.Timedelta(days=30)
            sales_last = df_temp[df_temp['date'] >= last30]['total_spent'].sum()
            sales_prev = df_temp[(df_temp['date'] >= prev30) & (df_temp['date'] < last30)]['total_spent'].sum()
            growth = ((sales_last - sales_prev) / sales_prev * 100) if sales_prev > 0 else 0
            col1, col2 = st.columns(2)
            col1.metric("💰 Ventas últimos 30 días", f"${sales_last:,.0f}")
            col2.metric("📈 Crecimiento", f"{growth:.1f}%")


# =====================================================
# FUNCIONES DE IA (Groq)
# =====================================================
def generate_ai_insights(df):
    api_key = st.session_state.get("groq_api_key")
    if not api_key:
        raise ValueError("No hay API Key de Groq")

    stats = df.describe().round(2).to_string()
    trends = []
    if 'total_spent' in df.columns:
        trends.append(f"Venta promedio: {df['total_spent'].mean():.2f}")
        trends.append(f"Venta máxima: {df['total_spent'].max():.2f}")
    if 'category' in df.columns:
        top_cat = df.groupby('category')['total_spent'].sum().idxmax()
        trends.append(f"Categoría dominante: {top_cat}")

    prompt = f"""
Eres analista de datos senior especializado en retail.

Estadísticas:
{stats}

Tendencias:
{chr(10).join(trends)}

Proporciona:
1. 3 insights principales del negocio
2. 2 riesgos potenciales
3. 3 recomendaciones estratégicas accionables
4. 1 pregunta estratégica para profundizar

Responde en español, tono ejecutivo, con bullet points.
"""
    url = "https://api.groq.com/openai/v1/chat/completions"
    payload = {
        "model": "mixtral-8x7b-32768",  # modelo estable y rápido
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.3
    }
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    response = requests.post(url, headers=headers, json=payload, timeout=30)
    if response.status_code != 200:
        raise RuntimeError(f"Error API: {response.text}")
    return response.json()["choices"][0]["message"]["content"]


def show_ai_insights_page():
    st.markdown('<h1 class="main-header">🤖 Insights Generados con IA</h1>', unsafe_allow_html=True)
    if st.session_state.df_clean is None:
        st.warning("⚠️ Primero debes cargar y limpiar los datos.")
        return
    if not st.session_state.groq_api_key:
        st.error("🔑 Ingresa tu API Key de Groq en la barra lateral.")
        return

    df = st.session_state.df_clean
    st.markdown("Esta sección usa **Groq + Mixtral** para generar insights automáticos.")
    with st.expander("⚙️ Configuración"):
        st.write("Modelo: **mixtral-8x7b-32768** (estable)")

    if st.button("🚀 Generar Insights con IA", type="primary"):
        with st.spinner("Analizando datos y consultando IA..."):
            try:
                insights = generate_ai_insights(df)
                st.success("✅ Insights generados")
                st.markdown("### 🧠 Análisis Ejecutivo")
                st.markdown(insights)
                st.session_state.ai_insights = insights
            except Exception as e:
                st.error(f"Error: {str(e)}")

    if "ai_insights" in st.session_state and st.session_state.ai_insights:
        st.markdown("---")
        st.subheader("📥 Exportar Insights")
        st.download_button("Descargar TXT", st.session_state.ai_insights,
                           "insights_ia.txt", "text/plain")


# =====================================================
# EJECUCIÓN PRINCIPAL
# =====================================================
if __name__ == "__main__":
    main()