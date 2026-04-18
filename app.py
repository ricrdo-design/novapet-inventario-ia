import os
import math
import joblib
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

# =========================================================
# CONFIGURACIÓN GENERAL
# =========================================================
st.set_page_config(
    page_title="NovaPet | Predicción de inventario",
    page_icon="📦",
    layout="wide"
)

# =========================================================
# ESTILO VISUAL
# =========================================================
st.markdown(
    """
    <style>
    .main {
        background-color: #0b1220;
        color: #f3f4f6;
    }

    .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
        max-width: 900px;
    }

    h1, h2, h3, h4 {
        color: #f8fafc;
    }

    p, li, label, .stMarkdown, .stCaption {
        color: #e5e7eb;
    }

    .custom-card {
        background-color: #152238;
        padding: 1.2rem;
        border-radius: 14px;
        margin-bottom: 1rem;
    }

    .success-card {
        background-color: #14532d;
        padding: 1rem 1.2rem;
        border-radius: 12px;
        margin-top: 0.8rem;
        margin-bottom: 0.8rem;
        color: white;
        font-weight: 600;
    }

    .warning-card {
        background-color: #5b4a0a;
        padding: 1rem 1.2rem;
        border-radius: 12px;
        margin-top: 0.8rem;
        margin-bottom: 0.8rem;
        color: white;
        font-weight: 600;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# =========================================================
# FUNCIONES AUXILIARES
# =========================================================
def redondear_hacia_arriba(valor: float) -> int:
    return int(math.ceil(valor))


def predecir_baseline(semana_4: float, semana_3: float, semana_2: float, semana_1: float) -> float:
    return (semana_4 + semana_3 + semana_2 + semana_1) / 4


def construir_input_modelo(
    producto: str,
    categoria: str,
    precio_unitario: float,
    semana_4: float,
    semana_3: float,
    semana_2: float,
    semana_1: float,
    anio: int,
    semana: int,
) -> pd.DataFrame:
    promedio_4 = (semana_4 + semana_3 + semana_2 + semana_1) / 4
    min_4 = min(semana_4, semana_3, semana_2, semana_1)
    max_4 = max(semana_4, semana_3, semana_2, semana_1)

    valores = [semana_4, semana_3, semana_2, semana_1]
    std_4 = float(pd.Series(valores).std()) if len(set(valores)) > 1 else 0.0

    tendencia_2 = semana_1 - semana_2

    df_input = pd.DataFrame(
        [
            {
                "producto": producto,
                "categoria": categoria,
                "anio": anio,
                "semana": semana,
                "precio_unitario": precio_unitario,
                "lag_1": semana_1,
                "lag_2": semana_2,
                "lag_3": semana_3,
                "lag_4": semana_4,
                "promedio_4": promedio_4,
                "min_4": min_4,
                "max_4": max_4,
                "std_4": std_4,
                "tendencia_2": tendencia_2,
            }
        ]
    )

    return df_input


@st.cache_resource
def cargar_modelo_rf():
    ruta = "modelo_rf_inventario_novapet.pkl"
    if os.path.exists(ruta):
        return joblib.load(ruta)
    return None


def render_card(texto: str, color_class: str = "custom-card"):
    st.markdown(f"<div class='{color_class}'>{texto}</div>", unsafe_allow_html=True)


# =========================================================
# CARGA MODELO EXPERIMENTAL
# =========================================================
modelo_rf = cargar_modelo_rf()

# =========================================================
# HEADER
# =========================================================
st.title("NovaPet | Predicción de inventario")
st.subheader("Medicamentos y vacunas")
st.caption("Prototipo funcional de apoyo a la decisión para estimar consumo semanal y recomendar compras.")

st.info(
    "Este prototipo utiliza como motor principal de predicción el **Baseline "
    "(promedio móvil de 4 semanas)**, debido a que fue el enfoque con mejor "
    "desempeño en la validación técnica del proyecto. "
    "**Random Forest** se mantiene como referencia experimental secundaria."
)

# =========================================================
# 1) VALIDACIÓN TÉCNICA DEL PROYECTO
# =========================================================
st.markdown("## 1) Validación técnica del proyecto")

df_validacion = pd.DataFrame(
    {
        "Modelo": [
            "Baseline (Promedio móvil 4 semanas)",
            "Random Forest",
            "XGBoost",
        ],
        "MAE": [1.797, 2.178, 2.559],
        "RMSE": [3.067, 4.139, 4.148],
        "R²": [0.583, 0.241, 0.238],
        "Mejora_MAE_vs_Baseline_%": [0.0, -21.2, -42.4],
    }
)

st.dataframe(df_validacion, use_container_width=True)

render_card(
    "Modelo con mejor validación: <b>Baseline (Promedio móvil 4 semanas)</b> | "
    "MAE = <b>1.797</b> | RMSE = <b>3.067</b> | R² = <b>0.583</b>",
    color_class="success-card"
)

st.caption(
    "Interpretación: un menor MAE y RMSE indica menor error de predicción. "
    "Entre los modelos de machine learning evaluados, **Random Forest** presentó "
    "mejor desempeño que **XGBoost**, aunque ambos fueron superados por el Baseline."
)

# =========================================================
# 2) DATOS DE ENTRADA
# =========================================================
st.markdown("## 2) Datos de entrada")
st.markdown("### Cómo ingresar los datos correctamente")

st.markdown(
    """
- **Semana -4:** consumo de hace 4 semanas  
- **Semana -3:** consumo de hace 3 semanas  
- **Semana -2:** consumo de hace 2 semanas  
- **Semana -1:** consumo de la última semana cerrada  

La predicción principal estima el **consumo esperado de la próxima semana**.  
Luego, el sistema calcula una recomendación de compra para cubrir:

- **4 semanas proyectadas**
- **+ 20% de stock de seguridad**
"""
)

col1, col2 = st.columns(2)
with col1:
    producto = st.text_input("Producto", value="MELOXICAM 1.5MG 10ML")
with col2:
    categoria = st.text_input("Categoría", value="FARMACIA")

precio_unitario = st.number_input(
    "Precio unitario promedio (USD)",
    min_value=0.0,
    value=1.70,
    step=0.10,
    format="%.2f",
)

col_anio, col_semana = st.columns(2)
with col_anio:
    anio = st.number_input("Año", min_value=2024, value=2026, step=1)
with col_semana:
    semana = st.number_input("Semana objetivo", min_value=1, max_value=53, value=13, step=1)

st.markdown("### Consumos recientes (unidades)")
c1, c2, c3, c4 = st.columns(4)

with c1:
    semana_4 = st.number_input("Semana -4", min_value=0.0, value=4.0, step=1.0)
with c2:
    semana_3 = st.number_input("Semana -3", min_value=0.0, value=3.0, step=1.0)
with c3:
    semana_2 = st.number_input("Semana -2", min_value=0.0, value=3.0, step=1.0)
with c4:
    semana_1 = st.number_input("Semana -1", min_value=0.0, value=6.0, step=1.0)

stock_actual = st.number_input("Stock actual (unidades)", min_value=0.0, value=4.0, step=1.0)

mostrar_rf = st.checkbox("Mostrar comparación experimental con Random Forest", value=True)

# =========================================================
# 3) PREDICCIÓN Y RECOMENDACIÓN
# =========================================================
st.markdown("## 3) Predicción y recomendación")

if st.button("Calcular recomendación"):
    pred_baseline = predecir_baseline(semana_4, semana_3, semana_2, semana_1)

    demanda_4_semanas = pred_baseline * 4
    stock_seguridad = demanda_4_semanas * 0.20
    stock_objetivo = demanda_4_semanas + stock_seguridad
    compra_recomendada = max(stock_objetivo - stock_actual, 0)
    compra_recomendada_redondeada = redondear_hacia_arriba(compra_recomendada)

    render_card(
        f"Consumo esperado (próxima semana) con <b>Baseline</b>: <b>{pred_baseline:.2f} unidades</b>",
        color_class="success-card"
    )

    st.markdown(f"**Demanda esperada 4 semanas:** {demanda_4_semanas:.2f} unidades")
    st.markdown(f"**Stock de seguridad (20%):** {stock_seguridad:.2f} unidades")
    st.markdown(f"**Stock objetivo:** {stock_objetivo:.2f} unidades")
    st.markdown(f"**Stock actual:** {stock_actual:.2f} unidades")

    render_card(
        f"Compra recomendada: <b>{compra_recomendada_redondeada} unidades</b> "
        f"(redondeado hacia arriba)",
        color_class="warning-card"
    )

    # =====================================================
    # 4) COMPARACIÓN EXPERIMENTAL
    # =====================================================
    pred_rf = None

    if mostrar_rf:
        st.markdown("## 4) Comparación experimental")

        if modelo_rf is None:
            st.warning(
                "No se encontró el archivo **modelo_rf_inventario_novapet.pkl**. "
                "La comparación experimental con Random Forest no se puede ejecutar en esta versión."
            )
        else:
            df_input_modelo = construir_input_modelo(
                producto=producto,
                categoria=categoria,
                precio_unitario=precio_unitario,
                semana_4=semana_4,
                semana_3=semana_3,
                semana_2=semana_2,
                semana_1=semana_1,
                anio=anio,
                semana=semana,
            )

            try:
                pred_rf = float(modelo_rf.predict(df_input_modelo)[0])

                st.write(f"**Predicción experimental con Random Forest:** {pred_rf:.2f} unidades")
                st.write(f"**Predicción principal con Baseline:** {pred_baseline:.2f} unidades")

                diferencia = pred_rf - pred_baseline
                st.caption(
                    f"Diferencia Random Forest - Baseline: {diferencia:.2f} unidades. "
                    "La decisión operativa del prototipo se mantiene en el Baseline, "
                    "por ser el mejor modelo en validación."
                )

                st.dataframe(df_input_modelo, use_container_width=True)

            except Exception as e:
                st.error(f"No fue posible ejecutar la predicción experimental con Random Forest: {e}")

    # =====================================================
    # 5) VISUALIZACIÓN
    # =====================================================
    st.markdown("## 5) Visualización")

    labels = ["Semana -4", "Semana -3", "Semana -2", "Semana -1", "Pred. Baseline"]
    values = [semana_4, semana_3, semana_2, semana_1, pred_baseline]

    if pred_rf is not None:
        labels.append("Pred. RF")
        values.append(pred_rf)

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(labels, values, marker="o")
    ax.set_title("Consumo reciente y predicciones")
    ax.set_ylabel("Unidades")
    ax.tick_params(axis="x", rotation=45)

    st.pyplot(fig)

else:
    st.caption("Ingresa los datos y presiona **Calcular recomendación** para generar la predicción.")
