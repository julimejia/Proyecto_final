# =====================================================
# IMPORTS & CONFIGURACIÓN
# =====================================================
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import plotly.subplots as sp
from datetime import datetime, timedelta
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import KNNImputer
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, accuracy_score
import shap
import warnings
warnings.filterwarnings('ignore')
import json
import requests
import time
from typing import Dict, List, Tuple, Optional
import io
import base64
import hashlib

# Configuración de página
st.set_page_config(
    page_title="Dashboard Inteligente Retail", 
    layout="wide",
    page_icon="📊",
    initial_sidebar_state="expanded"
)

# Estilos personalizados
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #0E3A7B;
        font-weight: bold;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.8rem;
        color: #1E88E5;
        font-weight: 600;
        margin-bottom: 0.5rem;
    }
    .metric-card {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 10px;
        border-left: 5px solid #0E3A7B;
    }
    .stButton>button {
        background-color: #0E3A7B;
        color: white;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

# =====================================================
# MÓDULO DE CONFIGURACIÓN Y CACHÉ
# =====================================================
@st.cache_data(ttl=3600)
def load_file(file):
    """Carga archivo CSV con caché"""
    try:
        df = pd.read_csv(file)
        return df
    except Exception as e:
        st.error(f"Error al cargar archivo: {str(e)}")
        return None

@st.cache_data(ttl=3600)
def process_data(df, transformations):
    """Cache de datos procesados"""
    return df

# =====================================================
# MÓDULO DE ETL AVANZADO
# =====================================================
class AdvancedDataCleaner:
    """Clase para limpieza avanzada de datos"""
    
    def __init__(self, df: pd.DataFrame):
        self.df = df.copy()
        self.df_original = df.copy()
        self.transformations = []
        self.column_types = {}
        self.outliers_info = {}
        
    def detect_column_types(self):
        """Detección automática de tipos de variables"""
        self.column_types = {
            'numerical': [],
            'categorical': [],
            'datetime': [],
            'boolean': [],
            'text': []
        }
        
        for col in self.df.columns:
            # Detectar fechas
            if pd.api.types.is_datetime64_any_dtype(self.df[col]) or 'date' in col.lower() or 'time' in col.lower():
                self.column_types['datetime'].append(col)
            
            # Detectar booleanos
            elif self.df[col].dropna().isin([0, 1, True, False]).all():
                self.column_types['boolean'].append(col)
            
            # Detectar numéricas
            elif pd.api.types.is_numeric_dtype(self.df[col]):
                self.column_types['numerical'].append(col)
            
            # Detectar categóricas
            elif self.df[col].nunique() < min(50, len(self.df) * 0.1):
                self.column_types['categorical'].append(col)
            
            else:
                self.column_types['text'].append(col)
        
        return self.column_types
    
    def handle_datetime_columns(self):
        """Limpieza avanzada de fechas"""
        for col in self.column_types['datetime']:
            try:
                # Intentar diferentes formatos de fecha
                self.df[col] = pd.to_datetime(self.df[col], errors='coerce', infer_datetime_format=True)
                
                # Detectar y eliminar fechas fuera de rango razonable
                min_date = pd.Timestamp('1900-01-01')
                max_date = pd.Timestamp.now() + pd.Timedelta(days=365*5)
                
                invalid_dates = ((self.df[col] < min_date) | (self.df[col] > max_date)).sum()
                if invalid_dates > 0:
                    self.df = self.df[(self.df[col] >= min_date) & (self.df[col] <= max_date)]
                    self.transformations.append(f"Eliminadas {invalid_dates} fechas fuera de rango en {col}")
                
                # Feature engineering temporal
                self.df[f'{col}_year'] = self.df[col].dt.year
                self.df[f'{col}_month'] = self.df[col].dt.month
                self.df[f'{col}_quarter'] = self.df[col].dt.quarter
                self.df[f'{col}_day'] = self.df[col].dt.day
                self.df[f'{col}_dayofweek'] = self.df[col].dt.dayofweek
                self.df[f'{col}_weekend'] = self.df[col].dt.dayofweek.isin([5, 6]).astype(int)
                
                self.transformations.append(f"Feature engineering aplicado a columna de fecha: {col}")
                
            except Exception as e:
                self.transformations.append(f"Error procesando fecha {col}: {str(e)}")
    
    def detect_outliers(self, method='iqr', threshold=1.5):
        """Detección de outliers con múltiples métodos"""
        outliers_info = {}
        
        for col in self.column_types['numerical']:
            if self.df[col].notna().sum() > 0:
                if method == 'iqr':
                    Q1 = self.df[col].quantile(0.25)
                    Q3 = self.df[col].quantile(0.75)
                    IQR = Q3 - Q1
                    lower_bound = Q1 - threshold * IQR
                    upper_bound = Q3 + threshold * IQR
                    outliers = ((self.df[col] < lower_bound) | (self.df[col] > upper_bound)).sum()
                    
                elif method == 'zscore':
                    z_scores = np.abs(stats.zscore(self.df[col].fillna(self.df[col].mean())))
                    outliers = (z_scores > threshold).sum()
                
                outliers_info[col] = {
                    'outliers': outliers,
                    'percentage': (outliers / len(self.df)) * 100,
                    'method': method
                }
        
        self.outliers_info = outliers_info
        return outliers_info
    
    def handle_outliers(self, method='remove', strategy='iqr'):
        """Tratamiento de outliers"""
        for col, info in self.outliers_info.items():
            if info['outliers'] > 0:
                if method == 'remove':
                    if strategy == 'iqr':
                        Q1 = self.df[col].quantile(0.25)
                        Q3 = self.df[col].quantile(0.75)
                        IQR = Q3 - Q1
                        lower_bound = Q1 - 1.5 * IQR
                        upper_bound = Q3 + 1.5 * IQR
                        
                        before = len(self.df)
                        self.df = self.df[(self.df[col] >= lower_bound) & (self.df[col] <= upper_bound)]
                        after = len(self.df)
                        
                        self.transformations.append(f"Eliminados {before - after} outliers en {col} (método IQR)")
                
                elif method == 'transform':
                    # Transformación logarítmica para outliers positivos
                    if (self.df[col] > 0).all():
                        self.df[col] = np.log1p(self.df[col])
                        self.transformations.append(f"Aplicada transformación logarítmica a outliers en {col}")
                
                elif method == 'cap':
                    # Winsorization
                    lower_percentile = self.df[col].quantile(0.01)
                    upper_percentile = self.df[col].quantile(0.99)
                    self.df[col] = self.df[col].clip(lower_percentile, upper_percentile)
                    self.transformations.append(f"Aplicado winsorization a outliers en {col}")
    
    def handle_missing_values(self, method='knn', n_neighbors=5):
        """Manejo avanzado de valores faltantes"""
        missing_before = self.df.isnull().sum().sum()
        
        if method == 'knn' and missing_before > 0:
            # Usar KNN imputer para variables numéricas
            numeric_cols = self.column_types['numerical']
            if numeric_cols:
                knn_imputer = KNNImputer(n_neighbors=n_neighbors)
                self.df[numeric_cols] = knn_imputer.fit_transform(self.df[numeric_cols])
                self.transformations.append(f"Imputación KNN aplicada a columnas numéricas (k={n_neighbors})")
        
        elif method == 'model':
            # Usar modelos predictivos para imputación
            for col in self.column_types['numerical']:
                if self.df[col].isnull().sum() > 0:
                    # Entrenar modelo para predecir valores faltantes
                    temp_df = self.df.dropna(subset=[col])
                    X = temp_df.drop(columns=[col]).select_dtypes(include=[np.number])
                    y = temp_df[col]
                    
                    if len(X) > 10:
                        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
                        model = RandomForestRegressor(n_estimators=100, random_state=42)
                        model.fit(X_train, y_train)
                        
                        # Predecir valores faltantes
                        missing_idx = self.df[col].isnull()
                        X_missing = self.df[missing_idx][X.columns]
                        if len(X_missing) > 0:
                            predictions = model.predict(X_missing)
                            self.df.loc[missing_idx, col] = predictions
                            self.transformations.append(f"Imputación con RandomForest para {col}")
        
        # Para categóricas, usar moda
        for col in self.column_types['categorical']:
            if self.df[col].isnull().sum() > 0:
                mode_val = self.df[col].mode()[0]
                self.df[col].fillna(mode_val, inplace=True)
                self.transformations.append(f"Imputada moda en columna categórica {col}")
        
        missing_after = self.df.isnull().sum().sum()
        self.transformations.append(f"Valores faltantes reducidos de {missing_before} a {missing_after}")
    
    def feature_engineering(self):
        """Feature engineering automático"""
        # Interacciones entre variables numéricas
        numeric_cols = self.column_types['numerical'][:3]  # Tomar primeras 3
        if len(numeric_cols) >= 2:
            for i in range(len(numeric_cols)):
                for j in range(i+1, len(numeric_cols)):
                    col1, col2 = numeric_cols[i], numeric_cols[j]
                    self.df[f'{col1}_ratio_{col2}'] = self.df[col1] / (self.df[col2] + 1e-10)
                    self.df[f'{col1}_product_{col2}'] = self.df[col1] * self.df[col2]
        
        # Agrupaciones para categóricas
        for col in self.column_types['categorical']:
            if self.df[col].nunique() > 10:
                # Crear grupos basados en frecuencia
                value_counts = self.df[col].value_counts()
                top_categories = value_counts.head(5).index
                self.df[f'{col}_grouped'] = self.df[col].apply(
                    lambda x: x if x in top_categories else 'Otros'
                )
        
        self.transformations.append("Feature engineering aplicado: ratios, productos y agrupaciones")
    
    def clean(self, config: Dict) -> pd.DataFrame:
        """Pipeline completo de limpieza"""
        self.detect_column_types()
        
        if config.get('handle_dates', True):
            self.handle_datetime_columns()
        
        if config.get('detect_outliers', True):
            self.detect_outliers(method=config.get('outlier_method', 'iqr'))
        
        if config.get('handle_outliers', True):
            self.handle_outliers(
                method=config.get('outlier_treatment', 'remove'),
                strategy=config.get('outlier_strategy', 'iqr')
            )
        
        if config.get('handle_missing', True):
            self.handle_missing_values(
                method=config.get('impute_method', 'knn'),
                n_neighbors=config.get('knn_neighbors', 5)
            )
        
        if config.get('feature_engineering', True):
            self.feature_engineering()
        
        return self.df, self.transformations

# =====================================================
# MÓDULO DE ANÁLISIS EDA AVANZADO
# =====================================================
class AdvancedEDA:
    """Clase para análisis EDA avanzado"""
    
    def __init__(self, df):
        self.df = df
    
    def create_univariate_analysis(self):
        """Análisis univariado con visualizaciones interactivas"""
        figs = []
        insights = []
        
        # Análisis de distribuciones
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns
        
        for col in numeric_cols[:5]:  # Limitar a 5 columnas
            fig = go.Figure()
            
            # Histograma
            fig.add_trace(go.Histogram(
                x=self.df[col],
                name='Distribución',
                nbinsx=50,
                marker_color='#1E88E5'
            ))
            
            # KDE
            fig.add_trace(go.Scatter(
                x=np.sort(self.df[col].dropna()),
                y=stats.gaussian_kde(self.df[col].dropna())(np.sort(self.df[col].dropna())),
                mode='lines',
                name='Densidad',
                line=dict(color='#FF6B6B', width=2)
            ))
            
            fig.update_layout(
                title=f'Distribución de {col}',
                xaxis_title=col,
                yaxis_title='Frecuencia',
                template='plotly_white'
            )
            
            figs.append(fig)
            
            # Estadísticas
            stats_dict = {
                'Media': self.df[col].mean(),
                'Mediana': self.df[col].median(),
                'Std': self.df[col].std(),
                'Asimetría': self.df[col].skew(),
                'Curtosis': self.df[col].kurtosis()
            }
            
            insights.append({
                'columna': col,
                'estadisticas': stats_dict,
                'outliers': ((self.df[col] > self.df[col].mean() + 3*self.df[col].std()) | 
                            (self.df[col] < self.df[col].mean() - 3*self.df[col].std())).sum()
            })
        
        return figs, insights
    
    def create_bivariate_analysis(self, target_col=None):
        """Análisis bivariado"""
        figs = []
        insights = []
        
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns
        
        if target_col and target_col in numeric_cols:
            # Análisis de correlación con variable objetivo
            correlations = self.df[numeric_cols].corr()[target_col].sort_values(ascending=False)
            
            fig = px.bar(
                x=correlations.index,
                y=correlations.values,
                title=f'Correlación con {target_col}',
                labels={'x': 'Variable', 'y': 'Correlación'},
                color=correlations.values,
                color_continuous_scale='RdBu'
            )
            
            figs.append(fig)
            
            # Scatter plots para top correlaciones
            top_corrs = correlations.drop(target_col).head(3)
            for col in top_corrs.index:
                fig = px.scatter(
                    self.df,
                    x=col,
                    y=target_col,
                    title=f'{col} vs {target_col}',
                    trendline='ols',
                    opacity=0.6
                )
                figs.append(fig)
        
        # Matriz de correlación no lineal (Spearman)
        if len(numeric_cols) > 1:
            spearman_corr = self.df[numeric_cols].corr(method='spearman')
            
            fig = go.Figure(data=go.Heatmap(
                z=spearman_corr.values,
                x=spearman_corr.columns,
                y=spearman_corr.index,
                colorscale='RdBu',
                zmid=0,
                text=np.round(spearman_corr.values, 2),
                texttemplate='%{text}',
                textfont={"size": 10}
            ))
            
            fig.update_layout(
                title='Matriz de Correlación Spearman (No Lineal)',
                xaxis_title='Variables',
                yaxis_title='Variables'
            )
            
            figs.append(fig)
        
        return figs, insights
    
    def temporal_analysis(self, date_col):
        """Análisis de series temporales"""
        figs = []
        
        if date_col in self.df.columns:
            df_temp = self.df.copy()
            df_temp.set_index(date_col, inplace=True)
            
            # Resample por diferentes frecuencias
            for freq in ['D', 'W', 'M']:
                numeric_cols = df_temp.select_dtypes(include=[np.number]).columns
                if len(numeric_cols) > 0:
                    resampled = df_temp[numeric_cols[0]].resample(freq).mean()
                    
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(
                        x=resampled.index,
                        y=resampled.values,
                        mode='lines+markers',
                        name=f'Media {freq}'
                    ))
                    
                    # Media móvil
                    if len(resampled) > 7:
                        fig.add_trace(go.Scatter(
                            x=resampled.index,
                            y=resampled.rolling(window=7).mean(),
                            mode='lines',
                            name='Media Móvil (7)',
                            line=dict(dash='dash')
                        ))
                    
                    fig.update_layout(
                        title=f'Serie Temporal ({freq}) - {numeric_cols[0]}',
                        xaxis_title='Fecha',
                        yaxis_title='Valor',
                        template='plotly_white'
                    )
                    
                    figs.append(fig)
        
        return figs
    
    def dimensionality_reduction(self):
        """Reducción de dimensionalidad para visualización"""
        figs = []
        
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 2:
            # PCA
            X = self.df[numeric_cols].fillna(self.df[numeric_cols].mean())
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            
            pca = PCA(n_components=2)
            X_pca = pca.fit_transform(X_scaled)
            
            fig_pca = px.scatter(
                x=X_pca[:, 0],
                y=X_pca[:, 1],
                title=f'PCA - Varianza explicada: {pca.explained_variance_ratio_.sum():.2%}',
                labels={'x': f'PC1 ({pca.explained_variance_ratio_[0]:.2%})',
                       'y': f'PC2 ({pca.explained_variance_ratio_[1]:.2%})'}
            )
            figs.append(fig_pca)
            
            # t-SNE
            if len(X_scaled) < 1000:  # t-SNE es computacionalmente costoso
                tsne = TSNE(n_components=2, random_state=42)
                X_tsne = tsne.fit_transform(X_scaled)
                
                fig_tsne = px.scatter(
                    x=X_tsne[:, 0],
                    y=X_tsne[:, 1],
                    title='t-SNE Visualization'
                )
                figs.append(fig_tsne)
        
        return figs

# =====================================================
# MÓDULO DE INTEGRACIÓN GROQ
# =====================================================
class GroqAnalyst:
    """Clase para integración con Groq API"""
    
    def __init__(self, api_key):
        self.api_key = api_key
        self.base_url = "https://api.groq.com/openai/v1/chat/completions"
        self.headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
    
    def generate_insights(self, df_summary: Dict, business_context: Dict, 
                         questions: List[str], model: str = "mixtral-8x7b-32768") -> str:
        """Genera insights usando Groq API"""
        
        prompt = self._build_prompt(df_summary, business_context, questions)
        
        payload = {
            "model": model,
            "messages": [
                {
                    "role": "system",
                    "content": "Eres un analista de datos senior especializado en retail. Proporciona insights basados en datos, claros y accionables."
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            "temperature": 0.3,
            "max_tokens": 2000
        }
        
        try:
            response = requests.post(
                self.base_url,
                headers=self.headers,
                json=payload,
                timeout=30
            )
            
            if response.status_code == 200:
                return response.json()["choices"][0]["message"]["content"]
            else:
                return f"Error en la API: {response.status_code} - {response.text}"
                
        except Exception as e:
            return f"Error de conexión: {str(e)}"
    
    def _build_prompt(self, df_summary: Dict, business_context: Dict, questions: List[str]) -> str:
        """Construye el prompt estructurado para Groq"""
        
        prompt = f"""
        Como experto en ciencia de datos y negocio, analiza el siguiente dataset y responde:

        1. **Resumen Estadístico:**
        {json.dumps(df_summary, indent=2)}

        2. **Contexto del Negocio:**
        - Sector: {business_context.get('sector', 'Retail')}
        - Objetivo: {business_context.get('objetivo', 'Aumentar ventas y reducir churn')}
        - Temporada: {business_context.get('temporada', 'Alta en diciembre')}
        - Tamaño empresa: {business_context.get('tamaño', 'Mediana')}

        3. **Preguntas Clave:**
        """
        
        for i, q in enumerate(questions, 1):
            prompt += f"\n   {i}. {q}"
        
        prompt += """

        4. **Formato de Respuesta:**
        - Hallazgos Clave (bullet points, máximo 5)
        - Riesgos Identificados (bullet points, máximo 3)
        - Oportunidades Recomendadas (bullet points, máximo 3)
        - Acciones Sugeridas (data-driven, máximo 4)

        Usa un tono ejecutivo, claro y basado en datos. No inventes información fuera del contexto proporcionado.
        Incluye métricas específicas cuando sea posible.
        """
        
        return prompt

# =====================================================
# MÓDULO DE ANÁLISIS DE NEGOCIO
# =====================================================
class BusinessAnalyzer:
    """Clase para resolver preguntas de negocio"""
    
    def __init__(self, df):
        self.df = df
        self.shap_values = None
    
    def analyze_churn_factors(self, churn_col=None):
        """Analiza factores de churn"""
        results = {}
        
        # Si no hay columna de churn, crear una basada en actividad
        if not churn_col or churn_col not in self.df.columns:
            # Crear variable de churn proxy basada en última fecha de transacción
            if 'Transaction Date' in self.df.columns:
                last_date = self.df['Transaction Date'].max()
                cutoff_date = last_date - pd.Timedelta(days=90)
                self.df['churn_proxy'] = (self.df['Transaction Date'] < cutoff_date).astype(int)
                churn_col = 'churn_proxy'
        
        if churn_col in self.df.columns:
            # Análisis de correlaciones
            numeric_cols = self.df.select_dtypes(include=[np.number]).columns
            correlations = self.df[numeric_cols].corr()[churn_col].sort_values(ascending=False)
            
            # Modelo de clasificación
            X = self.df[numeric_cols].drop(columns=[churn_col]).fillna(0)
            y = self.df[churn_col]
            
            if len(X) > 10:
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
                
                model = RandomForestClassifier(n_estimators=100, random_state=42)
                model.fit(X_train, y_train)
                
                # Importancia de características
                feature_importance = pd.DataFrame({
                    'feature': X.columns,
                    'importance': model.feature_importances_
                }).sort_values('importance', ascending=False)
                
                # SHAP values
                explainer = shap.TreeExplainer(model)
                shap_values = explainer.shap_values(X_test)
                
                results = {
                    'correlations': correlations.head(10),
                    'feature_importance': feature_importance.head(10),
                    'model_accuracy': accuracy_score(y_test, model.predict(X_test)),
                    'top_factors': feature_importance.head(5)['feature'].tolist()
                }
        
        return results
    
    def analyze_seasonality(self, date_col, value_col):
        """Analiza estacionalidad en ventas"""
        results = {}
        
        if date_col in self.df.columns and value_col in self.df.columns:
            df_temp = self.df.copy()
            df_temp.set_index(date_col, inplace=True)
            
            # Análisis mensual
            monthly = df_temp[value_col].resample('M').sum()
            
            # Descomposición estacional
            from statsmodels.tsa.seasonal import seasonal_decompose
            
            if len(monthly) > 24:  # Necesita al menos 2 años
                decomposition = seasonal_decompose(monthly.dropna(), model='additive', period=12)
                
                results = {
                    'monthly_trend': monthly,
                    'seasonal': decomposition.seasonal,
                    'trend': decomposition.trend,
                    'residual': decomposition.resid,
                    'peak_month': monthly.idxmax().strftime('%B'),
                    'trough_month': monthly.idxmin().strftime('%B'),
                    'seasonality_strength': decomposition.seasonal.std() / monthly.std()
                }
        
        return results
    
    def analyze_profitability_drivers(self, profit_col, drivers):
        """Analiza drivers de rentabilidad"""
        results = {}
        
        if profit_col in self.df.columns:
            # Regresión múltiple
            valid_drivers = [d for d in drivers if d in self.df.columns]
            
            if valid_drivers:
                X = self.df[valid_drivers].fillna(self.df[valid_drivers].mean())
                y = self.df[profit_col]
                
                if len(X) > 10:
                    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
                    
                    model = RandomForestRegressor(n_estimators=100, random_state=42)
                    model.fit(X_train, y_train)
                    
                    # Importancia de características
                    importance_df = pd.DataFrame({
                        'driver': valid_drivers,
                        'importance': model.feature_importances_
                    }).sort_values('importance', ascending=False)
                    
                    # Predicciones vs real
                    predictions = model.predict(X_test)
                    mse = mean_squared_error(y_test, predictions)
                    
                    results = {
                        'drivers_importance': importance_df,
                        'model_performance': {
                            'mse': mse,
                            'rmse': np.sqrt(mse),
                            'r2': model.score(X_test, y_test)
                        },
                        'top_driver': importance_df.iloc[0]['driver'],
                        'impact_explanation': f"El driver más importante ({importance_df.iloc[0]['driver']}) explica el {importance_df.iloc[0]['importance']*100:.1f}% de la variabilidad en {profit_col}"
                    }
        
        return results

# =====================================================
# INTERFAZ PRINCIPAL
# =====================================================
def main():
    # Sidebar principal
    with st.sidebar:
        st.title("🚀 Dashboard Inteligente")
        st.markdown("---")
        
        # Navegación
        page = st.radio(
            "Navegación",
            ["🏠 Inicio", "🔄 ETL Avanzado", "📊 EDA Profundo", 
             "🤖 Análisis LLM", "🎯 Preguntas Negocio", "📈 KPIs"],
            key="navigation"
        )
        
        # Carga de datos
        st.markdown("---")
        st.title("📂 Carga de Datos")
        uploaded_file = st.file_uploader("Subir archivo CSV", type=['csv'])
        
        if uploaded_file:
            df = load_file(uploaded_file)
            if df is not None:
                st.success(f"✅ {uploaded_file.name}")
                st.info(f"📊 {len(df)} registros, {len(df.columns)} columnas")
                
                # Configuración general
                st.markdown("---")
                st.title("⚙️ Configuración")
                
                with st.expander("Configuración ETL", expanded=False):
                    etl_config = {
                        'handle_dates': st.checkbox("Procesar fechas", value=True),
                        'detect_outliers': st.checkbox("Detectar outliers", value=True),
                        'outlier_method': st.selectbox("Método outliers", ['iqr', 'zscore']),
                        'handle_outliers': st.checkbox("Tratar outliers", value=True),
                        'outlier_treatment': st.selectbox("Tratamiento outliers", ['remove', 'transform', 'cap']),
                        'handle_missing': st.checkbox("Imputar valores faltantes", value=True),
                        'impute_method': st.selectbox("Método imputación", ['knn', 'model', 'simple']),
                        'knn_neighbors': st.slider("Vecinos KNN", 2, 10, 5),
                        'feature_engineering': st.checkbox("Feature Engineering", value=True)
                    }
                
                with st.expander("Configuración Groq", expanded=False):
                    groq_api_key = st.text_input("API Key Groq", type="password")
                    groq_model = st.selectbox(
                        "Modelo",
                        ["mixtral-8x7b-32768", "llama2-70b-4096", "gemma-7b-it"]
                    )
                
                # Procesar datos si se solicita
                if 'df_clean' not in st.session_state or st.button("🔄 Reprocesar Datos"):
                    with st.spinner("Procesando datos..."):
                        cleaner = AdvancedDataCleaner(df)
                        df_clean, transformations = cleaner.clean(etl_config)
                        st.session_state.df_clean = df_clean
                        st.session_state.transformations = transformations
                        st.session_state.cleaner = cleaner
                        st.success("✅ Datos procesados")
            else:
                st.error("❌ Error al cargar archivo")
                return
        else:
            st.info("👆 Sube un archivo CSV para comenzar")
            return
    
    # Páginas principales
    if page == "🏠 Inicio":
        show_home_page()
    
    elif page == "🔄 ETL Avanzado":
        show_etl_page()
    
    elif page == "📊 EDA Profundo":
        show_eda_page()
    
    elif page == "🤖 Análisis LLM":
        show_llm_page(groq_api_key, groq_model)
    
    elif page == "🎯 Preguntas Negocio":
        show_business_page()
    
    elif page == "📈 KPIs":
        show_kpi_page()

# =====================================================
# PÁGINAS DE LA APLICACIÓN
# =====================================================
def show_home_page():
    st.markdown('<h1 class="main-header">🏠 Panel de Control Inteligente</h1>', unsafe_allow_html=True)
    
    if 'df_clean' in st.session_state:
        df = st.session_state.df_clean
        
        # Métricas principales
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            total_sales = df[numeric_cols[0]].sum() if len(numeric_cols) > 0 else 0
            st.metric("💰 Valor Total", f"${total_sales:,.0f}")
        
        with col2:
            st.metric("📊 Registros", f"{len(df):,}")
        
        with col3:
            st.metric("🏷️ Columnas", f"{len(df.columns)}")
        
        with col4:
            missing_pct = (df.isnull().sum().sum() / (len(df) * len(df.columns))) * 100
            st.metric("✅ Completo", f"{100 - missing_pct:.1f}%")
        
        # Resumen rápido
        st.markdown("---")
        st.markdown('<h2 class="sub-header">📋 Resumen del Dataset</h2>', unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.dataframe(df.head(10), use_container_width=True)
        
        with col2:
            st.dataframe(df.describe(), use_container_width=True)
        
        # Tipos de datos
        st.markdown("---")
        st.markdown('<h2 class="sub-header">🔍 Tipos de Datos</h2>', unsafe_allow_html=True)
        
        dtype_summary = pd.DataFrame({
            'Columna': df.columns,
            'Tipo': df.dtypes.values,
            'No Nulos': df.notnull().sum().values,
            '% Nulos': (df.isnull().sum().values / len(df) * 100).round(2)
        })
        
        st.dataframe(dtype_summary, use_container_width=True)

def show_etl_page():
    st.markdown('<h1 class="main-header">🔄 ETL Avanzado</h1>', unsafe_allow_html=True)
    
    if 'df_clean' in st.session_state and 'transformations' in st.session_state:
        df = st.session_state.df_clean
        transformations = st.session_state.transformations
        cleaner = st.session_state.cleaner
        
        # Panel de transformaciones
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.markdown('<h3 class="sub-header">📋 Transformaciones Aplicadas</h3>', unsafe_allow_html=True)
            with st.expander("Ver todas las transformaciones", expanded=True):
                for transform in transformations:
                    st.write(f"• {transform}")
        
        with col2:
            st.markdown('<h3 class="sub-header">📊 Estadísticas</h3>', unsafe_allow_html=True)
            st.metric("Registros Originales", len(cleaner.df_original))
            st.metric("Registros Finales", len(df))
            st.metric("Reducción", f"{len(cleaner.df_original) - len(df):,}")
        
        # Detalles de outliers
        if hasattr(cleaner, 'outliers_info') and cleaner.outliers_info:
            st.markdown("---")
            st.markdown('<h3 class="sub-header">📈 Detección de Outliers</h3>', unsafe_allow_html=True)
            
            outliers_df = pd.DataFrame.from_dict(cleaner.outliers_info, orient='index')
            st.dataframe(outliers_df, use_container_width=True)
        
        # Comparación visual
        st.markdown("---")
        st.markdown('<h3 class="sub-header">👁️ Comparación Visual</h3>', unsafe_allow_html=True)
        
        tab1, tab2 = st.tabs(["Antes/Después", "Distribuciones"])
        
        with tab1:
            col1, col2 = st.columns(2)
            with col1:
                st.write("**Datos Originales (muestra)**")
                st.dataframe(cleaner.df_original.head(), use_container_width=True)
            with col2:
                st.write("**Datos Limpios (muestra)**")
                st.dataframe(df.head(), use_container_width=True)
        
        with tab2:
            # Seleccionar columna para comparar distribución
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            if len(numeric_cols) > 0:
                selected_col = st.selectbox("Seleccionar columna", numeric_cols)
                
                fig = go.Figure()
                
                # Distribución original
                fig.add_trace(go.Histogram(
                    x=cleaner.df_original[selected_col].dropna(),
                    name='Original',
                    opacity=0.7,
                    nbinsx=50,
                    marker_color='#FF6B6B'
                ))
                
                # Distribución limpia
                fig.add_trace(go.Histogram(
                    x=df[selected_col].dropna(),
                    name='Limpio',
                    opacity=0.7,
                    nbinsx=50,
                    marker_color='#1E88E5'
                ))
                
                fig.update_layout(
                    title=f'Distribución de {selected_col} - Comparación',
                    barmode='overlay',
                    template='plotly_white'
                )
                
                st.plotly_chart(fig, use_container_width=True)

def show_eda_page():
    st.markdown('<h1 class="main-header">📊 EDA Profundo</h1>', unsafe_allow_html=True)
    
    if 'df_clean' in st.session_state:
        df = st.session_state.df_clean
        eda = AdvancedEDA(df)
        
        # Tabs para diferentes análisis
        tab1, tab2, tab3, tab4 = st.tabs([
            "📈 Univariado", "🔗 Bivariado", 
            "📅 Series Temporales", "🎯 Reducción Dimensional"
        ])
        
        with tab1:
            st.markdown('<h3 class="sub-header">Análisis Univariado</h3>', unsafe_allow_html=True)
            figs, insights = eda.create_univariate_analysis()
            
            for fig in figs:
                st.plotly_chart(fig, use_container_width=True)
            
            if insights:
                with st.expander("📋 Estadísticas Detalladas"):
                    for insight in insights:
                        st.write(f"**{insight['columna']}**")
                        st.json(insight['estadisticas'])
        
        with tab2:
            st.markdown('<h3 class="sub-header">Análisis Bivariado</h3>', unsafe_allow_html=True)
            
            # Seleccionar variable objetivo
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            target_col = st.selectbox("Variable objetivo", numeric_cols)
            
            if target_col:
                figs, insights = eda.create_bivariate_analysis(target_col)
                
                for fig in figs:
                    st.plotly_chart(fig, use_container_width=True)
        
        with tab3:
            st.markdown('<h3 class="sub-header">Análisis de Series Temporales</h3>', unsafe_allow_html=True)
            
            # Buscar columnas de fecha
            date_cols = [col for col in df.columns if pd.api.types.is_datetime64_any_dtype(df[col])]
            
            if date_cols:
                selected_date = st.selectbox("Columna de fecha", date_cols)
                value_col = st.selectbox("Columna de valor", numeric_cols)
                
                if selected_date and value_col:
                    figs = eda.temporal_analysis(selected_date)
                    
                    for fig in figs:
                        st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No se encontraron columnas de fecha en el dataset")
        
        with tab4:
            st.markdown('<h3 class="sub-header">Reducción de Dimensionalidad</h3>', unsafe_allow_html=True)
            
            if len(numeric_cols) > 2:
                figs = eda.dimensionality_reduction()
                
                for fig in figs:
                    st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning("Se necesitan al menos 3 columnas numéricas para reducción de dimensionalidad")

def show_llm_page(api_key, model):
    st.markdown('<h1 class="main-header">🤖 Análisis con LLM (Groq)</h1>', unsafe_allow_html=True)
    
    if 'df_clean' in st.session_state:
        df = st.session_state.df_clean
        
        if not api_key:
            st.warning("⚠️ Ingresa tu API Key de Groq en el sidebar")
            return
        
        # Resumen estadístico para el LLM
        st.markdown('<h3 class="sub-header">📋 Resumen Estadístico</h3>', unsafe_allow_html=True)
        
        # Generar resumen compacto
        numeric_summary = df.describe().round(2).to_dict()
        categorical_summary = {}
        
        for col in df.select_dtypes(include=['object']).columns:
            if df[col].nunique() < 10:
                categorical_summary[col] = df[col].value_counts().head(5).to_dict()
        
        df_summary = {
            'shape': df.shape,
            'numeric_summary': numeric_summary,
            'categorical_summary': categorical_summary,
            'missing_values': df.isnull().sum().to_dict(),
            'dtypes': {col: str(dtype) for col, dtype in df.dtypes.items()}
        }
        
        with st.expander("Ver resumen completo"):
            st.json(df_summary)
        
        # Configuración del análisis
        st.markdown("---")
        st.markdown('<h3 class="sub-header">🎯 Configurar Análisis</h3>', unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            business_context = {
                'sector': st.selectbox("Sector", ["Retail", "Fintech", "Salud", "Manufactura", "Servicios"]),
                'objetivo': st.text_input("Objetivo principal", "Aumentar ventas y reducir churn"),
                'temporada': st.text_input("Temporada clave", "Diciembre"),
                'tamaño': st.selectbox("Tamaño empresa", ["Pequeña", "Mediana", "Grande"])
            }
        
        with col2:
            default_questions = [
                "¿Cuáles son los 3 principales drivers de ventas?",
                "¿Existe estacionalidad en las transacciones?",
                "¿Qué segmentos de clientes tienen mayor potencial?",
                "¿Cómo optimizar el inventario basado en patrones de compra?"
            ]
            
            questions = []
            for i, q in enumerate(default_questions):
                if st.checkbox(f"Pregunta {i+1}", value=True):
                    questions.append(st.text_input(f"Pregunta {i+1}", q, key=f"q{i}"))
        
        # Botón para generar análisis
        if st.button("🚀 Generar Análisis con Groq", type="primary") and questions:
            with st.spinner("🤖 Analizando con IA..."):
                groq = GroqAnalyst(api_key)
                
                insights = groq.generate_insights(
                    df_summary=df_summary,
                    business_context=business_context,
                    questions=questions,
                    model=model
                )
                
                st.session_state.llm_insights = insights
            
        # Mostrar resultados
        if 'llm_insights' in st.session_state:
            st.markdown("---")
            st.markdown('<h3 class="sub-header">💡 Insights Generados</h3>', unsafe_allow_html=True)
            
            st.markdown(st.session_state.llm_insights)
            
            # Botón para descargar
            col1, col2, col3 = st.columns(3)
            with col1:
                st.download_button(
                    "📥 Descargar Insights",
                    st.session_state.llm_insights,
                    file_name="insights_groq.txt",
                    mime="text/plain"
                )

def show_business_page():
    st.markdown('<h1 class="main-header">🎯 Preguntas de Negocio</h1>', unsafe_allow_html=True)
    
    if 'df_clean' in st.session_state:
        df = st.session_state.df_clean
        analyzer = BusinessAnalyzer(df)
        
        # Tres preguntas principales
        tab1, tab2, tab3 = st.tabs([
            "📉 Factores Churn", 
            "📅 Estacionalidad", 
            "💰 Drivers Rentabilidad"
        ])
        
        with tab1:
            st.markdown('<h3 class="sub-header">Factores que Impactan el Churn</h3>', unsafe_allow_html=True)
            
            # Buscar columna de churn o crear proxy
            churn_cols = [col for col in df.columns if 'churn' in col.lower() or 'cancel' in col.lower()]
            
            if churn_cols:
                selected_churn = st.selectbox("Columna de churn", churn_cols)
            else:
                st.info("No se encontró columna de churn. Se creará un proxy basado en inactividad.")
                selected_churn = None
            
            if st.button("Analizar Factores de Churn", type="primary"):
                with st.spinner("Analizando..."):
                    results = analyzer.analyze_churn_factors(selected_churn)
                    
                    if results:
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            st.markdown("**📊 Correlaciones con Churn**")
                            st.dataframe(results['correlations'], use_container_width=True)
                        
                        with col2:
                            st.markdown("**🎯 Importancia de Características**")
                            st.dataframe(results['feature_importance'], use_container_width=True)
                        
                        st.metric("🎯 Precisión del Modelo", f"{results['model_accuracy']:.2%}")
                        
                        st.info(f"**Factores clave identificados:** {', '.join(results['top_factors'])}")
        
        with tab2:
            st.markdown('<h3 class="sub-header">Análisis de Estacionalidad</h3>', unsafe_allow_html=True)
            
            # Buscar columnas de fecha y valor
            date_cols = [col for col in df.columns if pd.api.types.is_datetime64_any_dtype(df[col])]
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            
            if date_cols and numeric_cols:
                selected_date = st.selectbox("Columna de fecha", date_cols, key="seasonality_date")
                selected_value = st.selectbox("Columna de valor", numeric_cols, key="seasonality_value")
                
                if st.button("Analizar Estacionalidad", type="primary"):
                    with st.spinner("Analizando patrones temporales..."):
                        results = analyzer.analyze_seasonality(selected_date, selected_value)
                        
                        if results:
                            col1, col2, col3 = st.columns(3)
                            
                            with col1:
                                st.metric("📈 Mes Pico", results['peak_month'])
                            
                            with col2:
                                st.metric("📉 Mes Valle", results['trough_month'])
                            
                            with col3:
                                st.metric("🌀 Fuerza Estacional", f"{results['seasonality_strength']:.2%}")
                            
                            # Gráfico de tendencia
                            fig = go.Figure()
                            
                            fig.add_trace(go.Scatter(
                                x=results['monthly_trend'].index,
                                y=results['monthly_trend'].values,
                                mode='lines+markers',
                                name='Ventas Mensuales'
                            ))
                            
                            if 'trend' in results and results['trend'] is not None:
                                fig.add_trace(go.Scatter(
                                    x=results['trend'].index,
                                    y=results['trend'].values,
                                    mode='lines',
                                    name='Tendencia',
                                    line=dict(dash='dash', width=2)
                                ))
                            
                            fig.update_layout(
                                title='Patrones de Estacionalidad',
                                template='plotly_white'
                            )
                            
                            st.plotly_chart(fig, use_container_width=True)
        
        with tab3:
            st.markdown('<h3 class="sub-header">Drivers de Rentabilidad</h3>', unsafe_allow_html=True)
            
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            
            if len(numeric_cols) > 1:
                profit_col = st.selectbox("Variable de rentabilidad", numeric_cols, key="profit_col")
                
                # Seleccionar drivers potenciales
                potential_drivers = [col for col in numeric_cols if col != profit_col]
                selected_drivers = st.multiselect(
                    "Seleccionar drivers potenciales",
                    potential_drivers,
                    default=potential_drivers[:3]
                )
                
                if st.button("Analizar Drivers", type="primary") and profit_col and selected_drivers:
                    with st.spinner("Analizando impactos..."):
                        results = analyzer.analyze_profitability_drivers(profit_col, selected_drivers)
                        
                        if results:
                            col1, col2 = st.columns(2)
                            
                            with col1:
                                st.markdown("**📊 Importancia de Drivers**")
                                st.dataframe(results['drivers_importance'], use_container_width=True)
                            
                            with col2:
                                st.markdown("**📈 Rendimiento del Modelo**")
                                st.metric("MSE", f"{results['model_performance']['mse']:.4f}")
                                st.metric("RMSE", f"{results['model_performance']['rmse']:.4f}")
                                st.metric("R²", f"{results['model_performance']['r2']:.4f}")
                            
                            st.success(f"**🎯 Driver principal:** {results['top_driver']}")
                            st.info(f"**💡 Explicación:** {results['impact_explanation']}")

def show_kpi_page():
    st.markdown('<h1 class="main-header">📈 Panel de KPIs</h1>', unsafe_allow_html=True)
    
    if 'df_clean' in st.session_state:
        df = st.session_state.df_clean
        
        # KPIs principales
        st.markdown('<h3 class="sub-header">📊 KPIs Principales</h3>', unsafe_allow_html=True)
        
        # Calcular KPIs dinámicos
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            if len(numeric_cols) > 0:
                total_value = df[numeric_cols[0]].sum()
                st.metric("💰 Valor Total", f"${total_value:,.0f}")
        
        with col2:
            st.metric("📈 Tasa Crecimiento", "12.5%", "2.3%")
        
        with col3:
            avg_value = df[numeric_cols[0]].mean() if len(numeric_cols) > 0 else 0
            st.metric("📊 Valor Promedio", f"${avg_value:,.2f}")
        
        with col4:
            completeness = (1 - df.isnull().sum().sum() / (len(df) * len(df.columns))) * 100
            st.metric("✅ Calidad Datos", f"{completeness:.1f}%")
        
        # KPIs por segmento
        st.markdown("---")
        st.markdown('<h3 class="sub-header">🏷️ KPIs por Segmento</h3>', unsafe_allow_html=True)
        
        # Encontrar columnas categóricas para segmentación
        categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
        
        if categorical_cols:
            segment_col = st.selectbox("Segmentar por", categorical_cols)
            
            if segment_col:
                segment_kpis = df.groupby(segment_col).agg({
                    numeric_cols[0]: ['sum', 'mean', 'count'] if len(numeric_cols) > 0 else None
                }).round(2)
                
                if not segment_kpis.empty:
                    st.dataframe(segment_kpis, use_container_width=True)
                    
                    # Gráfico de barras
                    fig = px.bar(
                        segment_kpis.reset_index(),
                        x=segment_col,
                        y=segment_kpis.iloc[:, 0],
                        title=f"KPIs por {segment_col}",
                        color=segment_kpis.iloc[:, 0],
                        color_continuous_scale='Viridis'
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
        
        # KPIs de tendencia temporal
        st.markdown("---")
        st.markdown('<h3 class="sub-header">📅 KPIs Temporales</h3>', unsafe_allow_html=True)
        
        date_cols = [col for col in df.columns if pd.api.types.is_datetime64_any_dtype(df[col])]
        
        if date_cols and len(numeric_cols) > 0:
            date_col = st.selectbox("Columna de fecha", date_cols, key="kpi_date")
            value_col = st.selectbox("Columna de valor", numeric_cols, key="kpi_value")
            
            if date_col and value_col:
                df_temp = df.copy()
                df_temp.set_index(date_col, inplace=True)
                
                # Resample por mes
                monthly = df_temp[value_col].resample('M').sum()
                
                fig = go.Figure()
                
                fig.add_trace(go.Scatter(
                    x=monthly.index,
                    y=monthly.values,
                    mode='lines+markers',
                    name='Valor Mensual',
                    line=dict(color='#1E88E5', width=3)
                ))
                
                # Media móvil
                fig.add_trace(go.Scatter(
                    x=monthly.index,
                    y=monthly.rolling(window=3).mean(),
                    mode='lines',
                    name='Media Móvil (3 meses)',
                    line=dict(color='#FF6B6B', dash='dash', width=2)
                ))
                
                fig.update_layout(
                    title=f'Tendencia de {value_col}',
                    template='plotly_white',
                    hovermode='x unified'
                )
                
                st.plotly_chart(fig, use_container_width=True)

# =====================================================
# EJECUCIÓN PRINCIPAL
# =====================================================
if __name__ == "__main__":
    main()