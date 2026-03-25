import math
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st


# =========================================================
# CONFIGURACIÓN GENERAL
# =========================================================
st.set_page_config(
    page_title="NovaPet - Predicción de Inventario",
    page_icon="📦",
    layout="centered"
)

# =========================================================
# PARÁMETROS DEL PROTOTIPO
# =========================================================
HORIZON_WEEKS = 4
SAFETY_FACTOR = 0.20
XGB_MODEL_PATH = Path("modelo_xgb_inventario_novapet.pkl")

# Resultados de validación reales del proyecto
VALIDATION_RESULTS = pd.DataFrame(
    {
        "Modelo": [
            "Baseline (Promedio móvil 4 semanas)",
            "Random Forest",
            "XGBoost",
        ],
        "MAE": [1.797, 2.178, 2.559],
        "RMSE": [3.067, 4.139, 4.148],
        "R2": [0.583, 0.241, 0.238],
    }
)

VALIDATION_RESULTS["Mejora_MAE_vs_Baseline_%"] = (
    (VALIDATION_RESULTS.loc[0, "MAE"] - VALIDATION_RESULTS["MAE"])
    / VALIDATION_RESULTS.loc[0, "MAE"]
    * 100
).round(2)

VALIDATION_RESULTS.loc[
    VALIDATION_RESULTS["Modelo"] == "Baseline (Promedio móvil 4 semanas)",
    "Mejora_MAE_vs_Baseline_%"
] = 0.0


# =========================================================
# FUNCIONES AUXILIARES
# =========================================================
@st.cache_resource
def load_xgb_model(path: str):
    p = Path(path)
    if p.exists():
        return joblib.load(p)
    return None


def baseline_prediction(cons_w4: float, cons_w3: float, cons_w2: float, cons_w1: float) -> float:
    """
    Predicción principal del prototipo:
    promedio móvil de las últimas 4 semanas cerradas.
    """
    values = [cons_w4, cons_w3, cons_w2, cons_w1]
    return float(np.mean(values))


def build_xgb_input(
    producto: str,
    categoria: str,
    precio_unitario: float,
    cons_w4: float,
    cons_w3: float,
    cons_w2: float,
    cons_w1: float,
) -> pd.DataFrame:
    """
    Construye el DataFrame de entrada para el modelo XGBoost entrenado.
    Debe coincidir con las variables usadas en el notebook final.
    """
    from datetime import date

    today = date.today()
    iso = today.isocalendar()

    lag_1 = float(cons_w1)
    lag_2 = float(cons_w2)
    lag_3 = float(cons_w3)
    lag_4 = float(cons_w4)
    promedio_4 = float(np.mean([cons_w1, cons_w2, cons_w3, cons_w4]))

    X_pred = pd.DataFrame([{
        "producto": str(producto),
        "categoria": str(categoria),
        "anio": int(iso.year),
        "semana": int(iso.week),
        "precio_unitario": float(precio_unitario),
        "lag_1": lag_1,
        "lag_2": lag_2,
        "lag_3": lag_3,
        "lag_4": lag_4,
        "promedio_4": promedio_4,
    }])

    return X_pred


def compute_inventory_policy(pred_week: float, stock_actual: float):
    """
    Convierte la predicción semanal en recomendación de compra.
    """
    pred_week = max(0.0, float(pred_week))
    stock_actual = max(0.0, float(stock_actual))

    demanda_4_semanas = pred_week * HORIZON_WEEKS
    stock_seguridad = demanda_4_semanas * SAFETY_FACTOR
    stock_objetivo = demanda_4_semanas + stock_seguridad
    compra_recomendada = max(stock_objetivo - stock_actual, 0.0)

    return {
        "pred_week": pred_week,
        "demanda_4_semanas": demanda_4_semanas,
        "stock_seguridad": stock_seguridad,
        "stock_objetivo": stock_objetivo,
        "compra_recomendada": math.ceil(compra_recomendada)
    }


# =========================================================
# ENCABEZADO
# =========================================================
st.title("NovaPet | Predicción de inventario")
st.subheader("Medicamentos y vacunas")
st.caption(
    "Prototipo funcional de apoyo a la decisión para estimar consumo semanal y recomendar compras."
)

st.info(
    "Este prototipo utiliza como **motor principal de predicción** el "
    "**Baseline (promedio móvil de 4 semanas)**, debido a que fue el enfoque con mejor desempeño "
    "en la validación técnica del proyecto. XGBoost se mantiene como referencia experimental."
)

# =========================================================
# RESULTADOS DE VALIDACIÓN
# =========================================================
st.markdown("### 1) Validación técnica del proyecto")

st.dataframe(VALIDATION_RESULTS, use_container_width=True)

best_row = VALIDATION_RESULTS.sort_values("MAE").iloc[0]

st.success(
    f"Modelo con mejor validación: **{best_row['Modelo']}** | "
    f"MAE = **{best_row['MAE']:.3f}** | "
    f"RMSE = **{best_row['RMSE']:.3f}** | "
    f"R² = **{best_row['R2']:.3f}**"
)

st.caption(
    "Interpretación: un menor MAE y RMSE indica menor error de predicción. "
    "El Baseline superó a Random Forest y XGBoost con los datos reales actuales del Kardex."
)

# =========================================================
# ENTRADAS DEL USUARIO
# =========================================================
st.markdown("### 2) Datos de entrada")

st.markdown(
    """
**Cómo ingresar los datos correctamente**

- **Semana -4**: consumo de hace 4 semanas  
- **Semana -3**: consumo de hace 3 semanas  
- **Semana -2**: consumo de hace 2 semanas  
- **Semana -1**: consumo de la última semana cerrada  

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
    step=0.1,
    help="El baseline no depende del precio, pero se solicita para mantener trazabilidad y permitir comparación experimental con XGBoost."
)

st.markdown("#### Consumos recientes (unidades)")

c1, c2, c3, c4 = st.columns(4)

with c1:
    cons_w4 = st.number_input("Semana -4", min_value=0.0, value=4.0, step=1.0)

with c2:
    cons_w3 = st.number_input("Semana -3", min_value=0.0, value=3.0, step=1.0)

with c3:
    cons_w2 = st.number_input("Semana -2", min_value=0.0, value=3.0, step=1.0)

with c4:
    cons_w1 = st.number_input("Semana -1", min_value=0.0, value=6.0, step=1.0)

stock_actual = st.number_input(
    "Stock actual (unidades)",
    min_value=0.0,
    value=4.0,
    step=1.0
)

comparar_xgb = st.checkbox(
    "Mostrar comparación experimental con XGBoost",
    value=True
)

# Advertencia si no hay señal reciente
if (cons_w1 + cons_w2 + cons_w3 + cons_w4) == 0:
    st.warning(
        "No se ingresó consumo en las últimas 4 semanas. "
        "La predicción será poco representativa."
    )

# =========================================================
# CÁLCULO PRINCIPAL
# =========================================================
st.markdown("### 3) Predicción y recomendación")

if st.button("Calcular recomendación"):

    # -----------------------------
    # Predicción principal: Baseline
    # -----------------------------
    pred_baseline = baseline_prediction(cons_w4, cons_w3, cons_w2, cons_w1)
    result = compute_inventory_policy(pred_baseline, stock_actual)

    st.success(
        f"Consumo esperado (próxima semana) con **Baseline**: "
        f"**{result['pred_week']:.2f} unidades**"
    )

    st.write(
        f"**Demanda esperada {HORIZON_WEEKS} semanas:** {result['demanda_4_semanas']:.2f} unidades"
    )
    st.write(
        f"**Stock de seguridad ({int(SAFETY_FACTOR * 100)}%):** {result['stock_seguridad']:.2f} unidades"
    )
    st.write(
        f"**Stock objetivo:** {result['stock_objetivo']:.2f} unidades"
    )
    st.write(
        f"**Stock actual:** {stock_actual:.2f} unidades"
    )

    if result["compra_recomendada"] == 0:
        st.info("Compra recomendada: **0 unidades**")
    else:
        st.warning(
            f"Compra recomendada: **{result['compra_recomendada']} unidades** "
            f"(redondeado hacia arriba)"
        )

    # -----------------------------
    # Comparación experimental con XGBoost
    # -----------------------------
    if comparar_xgb:
        st.markdown("### 4) Comparación experimental")

        xgb_model = load_xgb_model(str(XGB_MODEL_PATH))

        if xgb_model is None:
            st.caption(
                "No se encontró el archivo `modelo_xgb_inventario_novapet.pkl`. "
                "La comparación experimental no se puede ejecutar en esta versión."
            )
        else:
            try:
                X_pred_xgb = build_xgb_input(
                    producto=producto,
                    categoria=categoria,
                    precio_unitario=precio_unitario,
                    cons_w4=cons_w4,
                    cons_w3=cons_w3,
                    cons_w2=cons_w2,
                    cons_w1=cons_w1,
                )

                pred_xgb = float(xgb_model.predict(X_pred_xgb)[0])
                pred_xgb = max(0.0, pred_xgb)

                st.write(f"**Predicción experimental con XGBoost:** {pred_xgb:.2f} unidades")
                st.write(f"**Predicción principal con Baseline:** {pred_baseline:.2f} unidades")

                diferencia = pred_xgb - pred_baseline
                st.caption(
                    f"Diferencia XGBoost - Baseline: {diferencia:.2f} unidades. "
                    "La decisión operativa del prototipo se mantiene en el Baseline, "
                    "por ser el mejor modelo en validación."
                )

                st.dataframe(X_pred_xgb, use_container_width=True)

            except Exception as e:
                st.error(f"No fue posible generar la comparación con XGBoost. Error: {e}")

    # -----------------------------
    # Gráfico
    # -----------------------------
    chart_df = pd.DataFrame({
        "Periodo": ["Semana -4", "Semana -3", "Semana -2", "Semana -1", "Pred. Baseline"],
        "Consumo": [cons_w4, cons_w3, cons_w2, cons_w1, pred_baseline]
    }).set_index("Periodo")

    st.markdown("### 5) Visualización")
    st.line_chart(chart_df)

    st.caption(
        "La línea final representa la predicción de la próxima semana usando el enfoque "
        "que obtuvo el mejor desempeño en la validación técnica."
    )

# =========================================================
# PIE
# =========================================================
st.divider()
st.caption(
    "NovaPet | Prototipo académico de predicción de inventario. "
    "El sistema implementa el enfoque de mejor desempeño validado y conserva XGBoost como comparación experimental."
)
