# =========================================================
# RETAIL STORE SALES – EDA + DATA CLEANING DASHBOARD
# =========================================================

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# =========================================================
# PAGE CONFIG
# =========================================================
st.set_page_config(page_title="Retail Sales EDA", layout="wide")
st.title("📊 Retail Store Sales — Data Quality & EDA Dashboard")
st.markdown("---")


# =========================================================
# CLEANING LOGGER
# =========================================================
class CleaningLogger:
    def __init__(self):
        self.steps = []

    def log(self, name, before, after, reason, method):
        self.steps.append({
            "Paso": len(self.steps) + 1,
            "Proceso": name,
            "Filas antes": before,
            "Filas después": after,
            "Eliminadas": before - after,
            "Razón": reason,
            "Método": method
        })

    def to_df(self):
        return pd.DataFrame(self.steps)


# =========================================================
# LOADER
# =========================================================

def cargar_retail_sales(file):

    df_raw = pd.read_csv(file)
    df = df_raw.copy()
    logger = CleaningLogger()

    # ----------------------------------------
    # Normalizar nombres de columnas
    df.columns = df.columns.str.strip()

    # ----------------------------------------
    # Duplicados
    before = len(df)
    df = df.drop_duplicates()
    logger.log(
        "Eliminar duplicados",
        before,
        len(df),
        "Registros repetidos generan sesgo",
        "drop_duplicates()"
    )

    # ----------------------------------------
    # Tipos numéricos
    numeric_cols = ["Price Per Unit", "Quantity", "Total Spent"]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # ----------------------------------------
    # Convertir fecha
    df["Transaction Date"] = pd.to_datetime(df["Transaction Date"], errors="coerce")

    before = len(df)
    df = df[df["Transaction Date"].notna()]
    logger.log(
        "Eliminar fechas inválidas",
        before,
        len(df),
        "Fechas corruptas no analizables",
        "dropna(Transaction Date)"
    )

    # ----------------------------------------
    # Valores negativos
    before = len(df)
    df = df[(df["Price Per Unit"] > 0) & (df["Quantity"] > 0)]
    logger.log(
        "Eliminar valores negativos",
        before,
        len(df),
        "Precio y cantidad deben ser positivos",
        "Price>0 & Quantity>0"
    )

    # ----------------------------------------
    # Recalcular total spent inconsistente
    df["Expected Total"] = df["Price Per Unit"] * df["Quantity"]

    before = len(df)
    df = df[np.isclose(df["Total Spent"], df["Expected Total"], atol=0.01)]
    logger.log(
        "Eliminar inconsistencias contables",
        before,
        len(df),
        "Total Spent inconsistente con Price*Quantity",
        "np.isclose()"
    )

    df.drop(columns=["Expected Total"], inplace=True)

    # ----------------------------------------
    # Outliers IQR (Price)
    Q1 = df["Price Per Unit"].quantile(0.25)
    Q3 = df["Price Per Unit"].quantile(0.75)
    IQR = Q3 - Q1

    before = len(df)
    df = df[(df["Price Per Unit"] >= Q1 - 1.5*IQR) &
            (df["Price Per Unit"] <= Q3 + 1.5*IQR)]

    logger.log(
        "Eliminar outliers de precio",
        before,
        len(df),
        "Precios extremos fuera del rango IQR",
        "IQR filtering"
    )

    # ----------------------------------------
    # Quantity outliers
    Q1 = df["Quantity"].quantile(0.25)
    Q3 = df["Quantity"].quantile(0.75)
    IQR = Q3 - Q1

    before = len(df)
    df = df[(df["Quantity"] >= Q1 - 1.5*IQR) &
            (df["Quantity"] <= Q3 + 1.5*IQR)]

    logger.log(
        "Eliminar outliers de cantidad",
        before,
        len(df),
        "Cantidades fuera del rango IQR",
        "IQR filtering"
    )

    # ----------------------------------------
    # Discount column limpieza
    df["Discount Applied"] = df["Discount Applied"].fillna(False).astype(bool)

    filas_eliminadas = len(df_raw) - len(df)

    return df_raw, df, filas_eliminadas, logger


# =========================================================
# FILE UPLOAD
# =========================================================
file = st.file_uploader("Upload Retail Sales CSV", type=["csv"])

if file:

    df_raw, df_clean, filas_eliminadas, logger = cargar_retail_sales(file)

    st.success(f"Dataset cargado correctamente — filas eliminadas: {filas_eliminadas}")

    # =========================================================
    # DATA OVERVIEW
    # =========================================================
    st.subheader("Dataset Overview")

    col1, col2, col3 = st.columns(3)

    col1.metric("Filas originales", len(df_raw))
    col2.metric("Filas limpias", len(df_clean))
    col3.metric("Columnas", len(df_clean.columns))

    st.dataframe(df_clean.head())

    # =========================================================
    # DATA QUALITY
    # =========================================================
    st.subheader("Data Quality Report")

    missing = df_clean.isna().sum().sort_values(ascending=False)
    st.bar_chart(missing)

    # =========================================================
    # DISTRIBUCIONES
    # =========================================================
    st.subheader("Distribución de Variables Numéricas")

    numeric_cols = df_clean.select_dtypes(include=np.number).columns

    for col in numeric_cols:
        fig, ax = plt.subplots()
        df_clean[col].hist(ax=ax, bins=40)
        ax.set_title(col)
        st.pyplot(fig)

    # =========================================================
    # SALES ANALYSIS
    # =========================================================
    st.subheader("Ventas por Categoría")

    sales_cat = df_clean.groupby("Category")["Total Spent"].sum().sort_values()
    st.bar_chart(sales_cat)

    # =========================================================
    # TIME ANALYSIS
    # =========================================================
    st.subheader("Ventas en el Tiempo")

    time_sales = df_clean.groupby(df_clean["Transaction Date"].dt.date)["Total Spent"].sum()
    st.line_chart(time_sales)

    # =========================================================
    # OUTLIER DETECTION
    # =========================================================
    st.subheader("Detección de Outliers (IQR)")

    for col in numeric_cols:

        Q1 = df_clean[col].quantile(0.25)
        Q3 = df_clean[col].quantile(0.75)
        IQR = Q3 - Q1

        outliers = df_clean[(df_clean[col] < Q1 - 1.5*IQR) | (df_clean[col] > Q3 + 1.5*IQR)]

        st.write(f"{col} — outliers detectados:", len(outliers))

    # =========================================================
    # CLEANING LOG
    # =========================================================
    st.subheader("Cleaning Log")

    st.dataframe(logger.to_df())
    # =========================================================
    # DATA HEALTH SCORE
    # =========================================================
    st.subheader("Data Health Score")

    missing_pct = (df_clean.isna().sum().sum()) / (df_clean.shape[0] * df_clean.shape[1])
    duplicate_pct = 1 - (len(df_clean.drop_duplicates()) / len(df_clean))

    health_score = int((1 - (missing_pct + duplicate_pct)/2) * 100)

    col1, col2, col3 = st.columns(3)
    col1.metric("Missing %", f"{missing_pct*100:.2f}%")
    col2.metric("Duplicate %", f"{duplicate_pct*100:.2f}%")
    col3.metric("Health Score", f"{health_score}/100")

    # =========================================================
    # CORRELATION MATRIX
    # =========================================================
    st.subheader("Correlation Matrix")

    corr = df_clean.select_dtypes(include=np.number).corr()

    fig, ax = plt.subplots()
    cax = ax.matshow(corr)
    fig.colorbar(cax)

    ax.set_xticks(range(len(corr.columns)))
    ax.set_xticklabels(corr.columns, rotation=90)
    ax.set_yticks(range(len(corr.columns)))
    ax.set_yticklabels(corr.columns)

    st.pyplot(fig)

    # =========================================================
    # BUSINESS KPIs
    # =========================================================
    st.subheader("Business KPIs")

    total_revenue = df_clean["Total Spent"].sum()
    avg_ticket = df_clean["Total Spent"].mean()
    total_transactions = len(df_clean)

    col1, col2, col3 = st.columns(3)
    col1.metric("Total Revenue", f"${total_revenue:,.0f}")
    col2.metric("Average Ticket", f"${avg_ticket:,.2f}")
    col3.metric("Transactions", total_transactions)

    # =========================================================
    # TOP PRODUCTS
    # =========================================================
    st.subheader("Top Products")

    top_items = df_clean.groupby("Item")["Total Spent"].sum().sort_values(ascending=False).head(10)
    st.bar_chart(top_items)

    # =========================================================
    # EXPORT CLEAN DATA
    # =========================================================
    st.subheader("Export Clean Dataset")

    csv = df_clean.to_csv(index=False).encode("utf-8")

    st.download_button(
        label="Download Clean CSV",
        data=csv,
        file_name="retail_sales_clean.csv",
        mime="text/csv"
    )
