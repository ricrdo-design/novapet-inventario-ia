import os
import math
from datetime import date, timedelta

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
# CATÁLOGO DE PRODUCTOS
# Ajusta precios si tienes valores reales actualizados
# =========================================================
CATALOGO_PRODUCTOS = {
    "MELOXICAM 1.5MG 10ML": {"categoria": "FARMACIA", "precio": 1.70},
    "RABISIN": {"categoria": "VACUNAS", "precio": 8.50},
    "ENTEROCHRONIC": {"categoria": "FARMACIA", "precio": 2.80},
    "PREVICOX 227MG TABLETA": {"categoria": "FARMACIA", "precio": 3.20},
    "VACUNA C6/CV": {"categoria": "VACUNAS", "precio": 7.50},
    "BRONCHICINE": {"categoria": "VACUNAS", "precio": 6.40},
    "TRITON LEVOTIROXINA 0.8MG": {"categoria": "FARMACIA", "precio": 4.10},
}


# =========================================================
# ESTILO VISUAL
# =========================================================
st.markdown(
    """
    <style>
    .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
        max-width: 1050px;
    }

    h1, h2, h3 {
        font-weight: 800;
    }

    .info-card {
        background-color: #16324f;
        padding: 1.1rem 1.3rem;
        border-radius: 14px;
        margin-bottom: 1rem;
        color: white;
        line-height: 1.5;
    }

    .success-card {
        background-color: #14532d;
        padding: 1rem 1.2rem;
        border-radius: 14px;
        margin-top: 0.8rem;
        margin-bottom: 0.8rem;
        color: white;
        font-weight: 700;
    }

    .warning-card {
        background-color: #665400;
        padding: 1rem 1.2rem;
        border-radius: 14px;
        margin-top: 0.8rem;
        margin-bottom: 0.8rem;
        color: white;
        font-weight: 700;
    }

    .danger-card {
        background-color: #7f1d1d;
        padding: 1rem 1.2rem;
        border-radius: 14px;
        margin-top: 0.8rem;
        margin-bottom: 0.8rem;
        color: white;
        font-weight: 700;
    }

    .small-note {
        color: #9ca3af;
        font-size: 0.9rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# =========================================================
# FUNCIONES AUXILIARES
# =========================================================
def redondear_hacia_arriba(valor: float) -> int:
    return int(math.ceil(valor))


def predecir_baseline(semana_4: int, semana_3: int, semana_2: int, semana_1: int) -> float:
    return (semana_4 + semana_3 + semana_2 + semana_1) / 4


def rango_semana_iso(anio: int, semana: int):
    inicio = date.fromisocalendar(int(anio), int(semana), 1)
    fin = inicio + timedelta(days=6)
    return inicio, fin


def formato_fecha_es(fecha: date) -> str:
    meses = {
        1: "enero",
        2: "febrero",
        3: "marzo",
        4: "abril",
        5: "mayo",
        6: "junio",
        7: "julio",
        8: "agosto",
        9: "septiembre",
        10: "octubre",
        11: "noviembre",
        12: "diciembre",
    }
    return f"{fecha.day} de {meses[fecha.month]} de {fecha.year}"


def construir_input_modelo(
    producto: str,
    categoria: str,
    precio_unitario: float,
    semana_4: int,
    semana_3: int,
    semana_2: int,
    semana_1: int,
    anio: int,
    semana: int,
) -> pd.DataFrame:
    promedio_4 = (semana_4 + semana_3 + semana_2 + semana_1) / 4
    min_4 = min(semana_4, semana_3, semana_2, semana_1)
    max_4 = max(semana_4, semana_3, semana_2, semana_1)

    valores = [semana_4, semana_3, semana_2, semana_1]
    std_4 = float(pd.Series(valores).std()) if len(set(valores)) > 1 else 0.0

    tendencia_2 = semana_1 - semana_2

    return pd.DataFrame(
        [
            {
                "producto": producto,
                "categoria": categoria,
                "anio": int(anio),
                "semana": int(semana),
                "precio_unitario": float(precio_unitario),
                "lag_1": int(semana_1),
                "lag_2": int(semana_2),
                "lag_3": int(semana_3),
                "lag_4": int(semana_4),
                "promedio_4": float(promedio_4),
                "min_4": int(min_4),
                "max_4": int(max_4),
                "std_4": float(std_4),
                "tendencia_2": int(tendencia_2),
            }
        ]
    )


@st.cache_resource
def cargar_modelo_rf():
    ruta = "modelo_rf_inventario_novapet.pkl"
    if os.path.exists(ruta):
        return joblib.load(ruta)
    return None


def render_card(texto: str, color_class: str):
    st.markdown(f"<div class='{color_class}'>{texto}</div>", unsafe_allow_html=True)


# =========================================================
# CARGA MODELO RF
# =========================================================
modelo_rf = cargar_modelo_rf()


# =========================================================
# HEADER
# =========================================================
st.title("NovaPet | Predicción de inventario")
st.subheader("Medicamentos y vacunas")
st.caption("Prototipo funcional de apoyo a la decisión para estimar consumo semanal y recomendar compras.")

st.markdown(
    """
    <div class="info-card">
    Este prototipo utiliza como motor principal de predicción el <b>Baseline 
    (promedio móvil de 4 semanas)</b>, debido a que fue el enfoque con mejor 
    desempeño en la validación técnica del proyecto. <b>Random Forest</b> se mantiene 
    como referencia experimental secundaria.
    </div>
    """,
    unsafe_allow_html=True,
)

if st.button("Limpiar valores"):
    st.session_state.clear()
    st.rerun()


# =========================================================
# 1) VALIDACIÓN TÉCNICA
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
    "success-card",
)

st.caption(
    "Interpretación: un menor MAE y RMSE indica menor error de predicción. "
    "Entre los modelos de machine learning evaluados, Random Forest presentó mejor desempeño "
    "que XGBoost, aunque ambos fueron superados por el Baseline."
)


# =========================================================
# 2) DATOS DE ENTRADA
# =========================================================
st.markdown("## 2) Datos de entrada")
st.markdown("### Producto a evaluar")

producto = st.selectbox(
    "Selecciona el producto",
    options=list(CATALOGO_PRODUCTOS.keys()),
    index=0,
    help="Selecciona el medicamento o vacuna desde el catálogo para evitar errores de digitación.",
)

categoria = CATALOGO_PRODUCTOS[producto]["categoria"]
precio_unitario = CATALOGO_PRODUCTOS[producto]["precio"]

col_cat, col_precio = st.columns(2)

with col_cat:
    st.text_input("Categoría", value=categoria, disabled=True)

with col_precio:
    st.number_input(
        "Precio unitario promedio (USD)",
        min_value=0.0,
        value=float(precio_unitario),
        step=0.10,
        format="%.2f",
        disabled=True,
    )

st.markdown("### Semana proyectada")

col_anio, col_semana = st.columns(2)

with col_anio:
    anio = st.number_input("Año", min_value=2024, max_value=2030, value=2026, step=1)

with col_semana:
    semana = st.number_input(
        "Semana a predecir",
        min_value=1,
        max_value=53,
        value=13,
        step=1,
        help="Semana calendario ISO para la cual se desea proyectar el consumo.",
    )

try:
    fecha_inicio, fecha_fin = rango_semana_iso(anio, semana)
    st.info(
        f"Semana proyectada: **lunes {formato_fecha_es(fecha_inicio)} – "
        f"domingo {formato_fecha_es(fecha_fin)}**"
    )
except Exception:
    st.warning("La semana seleccionada no es válida para el año indicado.")

st.markdown("### Consumos recientes")

st.markdown(
    """
    <span class="small-note">
    Ingresa el consumo real de las últimas cuatro semanas cerradas. 
    La predicción principal estima el consumo esperado de la siguiente semana.
    </span>
    """,
    unsafe_allow_html=True,
)

c1, c2, c3, c4 = st.columns(4)

with c1:
    semana_4 = st.number_input("Semana -4", min_value=0, value=4, step=1)
with c2:
    semana_3 = st.number_input("Semana -3", min_value=0, value=3, step=1)
with c3:
    semana_2 = st.number_input("Semana -2", min_value=0, value=3, step=1)
with c4:
    semana_1 = st.number_input("Semana -1", min_value=0, value=6, step=1)

stock_actual = st.number_input(
    "Stock actual disponible (unidades)",
    min_value=0,
    value=4,
    step=1,
)

stock_seguridad_pct = st.slider(
    "Stock de seguridad",
    min_value=0,
    max_value=50,
    value=20,
    step=5,
    help="Porcentaje adicional sobre la demanda proyectada para reducir riesgo de quiebre de stock.",
)

mostrar_rf = st.checkbox("Mostrar comparación experimental con Random Forest", value=True)


# =========================================================
# 3) PREDICCIÓN Y RECOMENDACIÓN
# =========================================================
st.markdown("## 3) Predicción y recomendación")

if st.button("Calcular recomendación", type="primary"):
    pred_baseline = predecir_baseline(semana_4, semana_3, semana_2, semana_1)

    consumo_semana = redondear_hacia_arriba(pred_baseline)
    demanda_4_semanas = consumo_semana * 4
    stock_seguridad = redondear_hacia_arriba(demanda_4_semanas * (stock_seguridad_pct / 100))
    stock_objetivo = demanda_4_semanas + stock_seguridad
    compra_recomendada_redondeada = max(stock_objetivo - int(stock_actual), 0)

    st.markdown("### Resultado operativo")

    k1, k2, k3, k4 = st.columns(4)

    k1.metric("Consumo próxima semana", f"{consumo_semana} u.")
    k2.metric("Demanda 4 semanas", f"{demanda_4_semanas} u.")
    k3.metric("Stock objetivo", f"{stock_objetivo} u.")
    k4.metric("Compra sugerida", f"{compra_recomendada_redondeada} u.")

    if stock_actual < consumo_semana:
        st.error("Riesgo alto: el stock actual no cubre el consumo esperado de la próxima semana.")
    else:
        st.success("Stock suficiente para cubrir el consumo esperado de la próxima semana.")

    if compra_recomendada_redondeada > 0:
        costo_compra = compra_recomendada_redondeada * precio_unitario
        st.info(f"Inversión estimada de compra: **${costo_compra:.2f} USD**")
    else:
        st.info("Inversión estimada de compra: **$0.00 USD**")

    if compra_recomendada_redondeada == 0:
        render_card("Estado: stock suficiente. No se recomienda compra inmediata.", "success-card")
    elif compra_recomendada_redondeada <= 5:
        render_card("Estado: compra moderada recomendada.", "warning-card")
    else:
        render_card(
            "Estado: compra prioritaria recomendada para evitar quiebre de stock.",
            "danger-card",
        )

    st.markdown(
        f"""
        **Detalle del cálculo:**  
        - Consumo esperado próxima semana: **{consumo_semana} unidades**  
        - Demanda proyectada a 4 semanas: **{demanda_4_semanas} unidades**  
        - Stock de seguridad ({stock_seguridad_pct}%): **{stock_seguridad} unidades**  
        - Stock objetivo: **{stock_objetivo} unidades**  
        - Stock actual: **{int(stock_actual)} unidades**  
        """
    )

    # =====================================================
    # 4) COMPARACIÓN EXPERIMENTAL RANDOM FOREST
    # =====================================================
    pred_rf = None

    if mostrar_rf:
        st.markdown("## 4) Comparación experimental")

        if modelo_rf is None:
            st.warning(
                "No se encontró el archivo **modelo_rf_inventario_novapet.pkl**. "
                "La comparación experimental con Random Forest no se puede ejecutar."
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
                pred_rf = redondear_hacia_arriba(float(modelo_rf.predict(df_input_modelo)[0]))

                col_rf1, col_rf2 = st.columns(2)
                col_rf1.metric("Predicción Baseline", f"{consumo_semana} u.")
                col_rf2.metric("Predicción Random Forest", f"{pred_rf} u.")

                diferencia = pred_rf - consumo_semana
                st.caption(
                    f"Diferencia Random Forest - Baseline: {diferencia} unidades. "
                    "La decisión operativa del prototipo se mantiene en el Baseline, "
                    "por ser el mejor modelo en validación."
                )

                with st.expander("Ver detalle técnico enviado al modelo Random Forest"):
                    st.dataframe(df_input_modelo, use_container_width=True)

            except Exception as e:
                st.error(f"No fue posible ejecutar la predicción experimental con Random Forest: {e}")

    # =====================================================
    # 5) VISUALIZACIÓN
    # =====================================================
    st.markdown("## 5) Visualización")

    labels = ["Semana -4", "Semana -3", "Semana -2", "Semana -1", "Pred. Baseline"]
    values = [semana_4, semana_3, semana_2, semana_1, consumo_semana]

    if pred_rf is not None:
        labels.append("Pred. RF")
        values.append(pred_rf)

    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.plot(labels, values, marker="o", linewidth=2)
    ax.set_title("Consumo reciente y predicciones")
    ax.set_ylabel("Unidades")
    ax.tick_params(axis="x", rotation=35)
    ax.grid(True, alpha=0.25)

    st.pyplot(fig)

else:
    st.caption("Ingresa los datos y presiona **Calcular recomendación** para generar la predicción.")
