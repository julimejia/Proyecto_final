
# 0 link del proyecto en streamlit  : https://proyectofinal-3geswd8ggswv3fsu4ekagz.streamlit.app/

```markdown
# Dashboard Retail Inteligente

Este proyecto es un dashboard interactivo construido con **Streamlit** que permite cargar, limpiar y analizar datos de ventas retail. Está diseñado para responder tres preguntas clave de negocio:

1. **Rentabilidad por categoría**: ¿Qué categorías generan mayor ingreso y cuáles tienen menor rentabilidad?
2. **Segmentos de clientes**: ¿Qué segmentos (ubicación, método de pago, categoría) tienen el ticket promedio más alto y cuál es su gasto total?
3. **Patrones temporales**: ¿Existen patrones semanales, mensuales u horarios en las ventas?

Además, incluye una sección de **insights con IA** mediante la API de Groq (modelo `llama-3.3-70b-versatile`) para generar recomendaciones automáticas.

---

## 🚀 Características

- **Carga de datos**: Sube archivos CSV con datos de ventas.
- **Limpieza automática**: Normalización de nombres, manejo de nulos, conversión de tipos y feature engineering temporal.
- **ETL y comparativa**: Visualización del antes/después y exportación de datos limpios.
- **Análisis de negocio**: Gráficos interactivos y tablas para cada pregunta.
- **EDA completo**: Distribuciones, correlaciones, series temporales y reporte ejecutivo.
- **KPIs**: Métricas principales, segmentación y comparativas temporales.
- **Integración con Groq**: Generación de insights ejecutivos mediante IA.

---

## 📋 Requisitos

- Python 3.9 o superior
- pip (gestor de paquetes)
- (Opcional) Una **API Key de Groq** para usar la sección de IA. Puedes obtenerla gratis en [console.groq.com](https://console.groq.com).

---

## 🛠️ Instalación y ejecución local

### 1. Clonar el repositorio

```bash
git clone [https://github.com/tu-usuario/dashboard-retail-inteligente.git](https://github.com/julimejia/Proyecto_final.git)
cd dashboard-retail-inteligente
```

### 2. Crear y activar un entorno virtual (recomendado)

**Windows:**
```bash
python -m venv venv
venv\Scripts\activate
```

**macOS / Linux:**
```bash
python3 -m venv venv
source venv/bin/activate
```

### 3. Instalar dependencias

```bash
pip install -r requirements.txt
```

**Contenido de `requirements.txt`:**
```
streamlit
pandas
numpy
plotly
requests
```

### 4. Ejecutar la aplicación

```bash
streamlit run app.py
```

La aplicación se abrirá automáticamente en tu navegador por defecto (normalmente en `http://localhost:8501`).

---

## 📁 Estructura del proyecto

```
dashboard-retail-inteligente/
│
├── app.py                  # Código principal de la aplicación
├── requirements.txt        # Dependencias
├── README.md               # Este archivo      
└── datasets/    # Carpeta para guardar datos de ejemplo
```

---

## 🔑 Configuración de la API Key de Groq (para IA)

1. Obtén tu API Key en [console.groq.com](https://console.groq.com).
2. En la barra lateral del dashboard, desplázate hasta la sección **"🤖 Configuración IA"**.
3. Pega tu clave en el campo de texto (se almacena solo en la sesión actual, no se guarda).

Una vez configurada, podrás usar la pestaña **"🤖 Insights IA"** para generar análisis automáticos.

---

Un dataset recomendado para pruebas es el [Retail Store Sales (dirty) de Kaggle](https://www.kaggle.com/datasets/ahmedmohamed2003/retail-store-sales-dirty-for-data-cleaning). El dashboard incluye una limpieza automática adaptada a este formato.

## 📄 Licencia

Este proyecto fue desarrollado con fines académicos para el curso **Fundamentos en Ciencia de Datos** de la **Universidad EAFIT**. Queda bajo la licencia [MIT](LICENSE).

---

## 👨‍💻 Autores

- **Juan Andrés Montoya**
- **Julián David Mejía**


🙏 CRÉDITOS Y AGRADECIMIENTOS

Este proyecto no habría sido posible sin el valioso trabajo y la inspiración de la comunidad de Kaggle.

- Notebook de referencia: El proceso de limpieza de datos y feature engineering está basado en el enfoque desarrollado en el notebook "Karnyxel is trying to clean the the dataset" (https://www.kaggle.com/code/kashifali68/karnyxel-is-trying-to-clean-the-the-dataset).

- Autor del código: Un agradecimiento especial a Kashif Ali (kashifali68) por publicar y compartir este detallado tutorial.

- Inspiración original: Extendemos nuestro reconocimiento a Karnyxel, cuya metodología y paso a paso para la limpieza de datos retail fueron seguidos e implementados en este dashboard.

Gracias por compartir conocimiento y contribuir al crecimiento de la comunidad. 🚀

Periodo 2026-1

---



