import datetime as dt
import math

import joblib
import numpy as np
import pandas as pd
import streamlit as st


# -----------------------------
# Configuración de la app
# -----------------------------
st.set_page_config(page_title="NovaPet - Predicción de Inventario", layout="centered")
st.title("NovaPet | Predicción de inventario (Medicamentos y Vacunas)")
st.caption("Modelo: XGBoost | Alcance: productos con historial de salidas en Kardex.")

# -----------------------------
# Política de reposición (MVP)
# -----------------------------
HORIZON_WEEKS = 4       # Cobertura objetivo (semanas)
SAFETY_FACTOR = 0.20    # 20% de stock de seguridad sobre la demanda del horizonte


@st.cache_resource
def load_model(path: str):
    return joblib.load(path)


def compute_recommendation(pred_week: float, stock_actual: float,
                           horizon_weeks: int = HORIZON_WEEKS,
                           safety_factor: float = SAFETY_FACTOR):
    """
    Calcula compra recomendada para cubrir 'horizon_weeks' semanas con stock de seguridad.
    - pred_week: consumo esperado semanal (salida del modelo)
    - stock_actual: stock disponible hoy
    """
    pred_week = max(0.0, float(pred_week))
    stock_actual = max(0.0, float(stock_actual))

    demand_h = pred_week * horizon_weeks            # demanda esperada en el horizonte
    safety_stock = safety_factor * demand_h         # stock de seguridad (simple)
    target_stock = demand_h + safety_stock          # stock objetivo total

    buy_units = max(0, math.ceil(target_stock - stock_actual))

    return demand_h, safety_stock, target_stock, buy_units


# -----------------------------
# Carga del modelo
# -----------------------------
model = load_model("modelo_xgb_inventario.pkl")

# Extraer categorías válidas (evita inputs inválidos)
pre = model.named_steps["preprocess"]
ohe = pre.named_transformers_["cat"]
productos_validos = list(ohe.categories_[0])
categorias_validas = list(ohe.categories_[1])

# -----------------------------
# Inputs
# -----------------------------
st.subheader("1) Datos de entrada")

col1, col2 = st.columns(2)
with col1:
    producto = st.selectbox("Producto", productos_validos)
with col2:
    categoria = st.selectbox("Categoría", categorias_validas)

precio_unitario = st.number_input(
    "Precio unitario promedio (USD)",
    min_value=0.0, value=1.0, step=0.1
)

st.markdown("**Consumos recientes (unidades)**")
MAX_CONSUMO = 500.0

c1, c2, c3, c4 = st.columns(4)
with c1:
    cons_w4 = st.number_input("Semana -4", min_value=0.0, max_value=MAX_CONSUMO, value=0.0, step=1.0)
with c2:
    cons_w3 = st.number_input("Semana -3", min_value=0.0, max_value=MAX_CONSUMO, value=0.0, step=1.0)
with c3:
    cons_w2 = st.number_input("Semana -2", min_value=0.0, max_value=MAX_CONSUMO, value=0.0, step=1.0)
with c4:
    cons_w1 = st.number_input("Semana -1", min_value=0.0, max_value=MAX_CONSUMO, value=0.0, step=1.0)

stock_actual = st.number_input(
    "Stock actual (unidades)",
    min_value=0.0, value=0.0, step=1.0
)

# Variables temporales: semana/año actual
today = dt.date.today()
iso = today.isocalendar()
anio = iso.year
semana = iso.week

# Features derivadas (según tu entrenamiento)
lag_1 = float(cons_w1)
lag_2 = float(cons_w2)
promedio_4 = float(np.mean([cons_w1, cons_w2, cons_w3, cons_w4]))

# DataFrame para predicción (debe coincidir con el pipeline entrenado)
X_pred = pd.DataFrame([{
    "producto": producto,
    "categoria": categoria,
    "anio": int(anio),
    "semana": int(semana),
    "precio_unitario": float(precio_unitario),
    "lag_1": lag_1,
    "lag_2": lag_2,
    "promedio_4": promedio_4,
}])

st.caption("Transparencia: estas son las variables que el modelo utiliza.")
st.dataframe(X_pred, use_container_width=True)

# Advertencia si no hay señal histórica
if (cons_w1 + cons_w2 + cons_w3 + cons_w4) == 0:
    st.warning(
        "No se ingresó consumo en las últimas 4 semanas. "
        "La predicción puede ser poco representativa; se recomienda ingresar consumos reales recientes."
    )

st.divider()
st.subheader("2) Predicción y recomendación")

# Explicación breve y clara de la política (para utilidad real)
st.info(
    f"**Política de reposición (MVP):** cobertura objetivo de **{HORIZON_WEEKS} semanas** "
    f"+ **{int(SAFETY_FACTOR*100)}%** de stock de seguridad. "
    "La compra recomendada se calcula para alcanzar el stock objetivo."
)

if st.button("Calcular recomendación"):
    # 1) Predicción semanal (salida del modelo)
    pred_consumo = float(model.predict(X_pred)[0])
    pred_consumo = max(pred_consumo, 0.0)  # asegurar no negativo

    # 2) Convertir predicción semanal a decisión operativa (4 semanas + seguridad)
    demand_4w, safety_stock, target_stock, compra_show = compute_recommendation(
        pred_week=pred_consumo,
        stock_actual=float(stock_actual),
        horizon_weeks=HORIZON_WEEKS,
        safety_factor=SAFETY_FACTOR
    )

    # 3) Mostrar resultados
    st.success(f"Consumo esperado (próxima semana): **{pred_consumo:.2f} unidades**")

    st.write(
        f"Demanda esperada {HORIZON_WEEKS} semanas: **{demand_4w:.2f}** unidades  \n"
        f"Stock de seguridad ({int(SAFETY_FACTOR*100)}%): **{safety_stock:.2f}** unidades  \n"
        f"Stock objetivo: **{target_stock:.2f}** unidades  \n"
        f"Stock actual: **{float(stock_actual):.2f}** unidades"
    )

    # Mensaje principal de compra
    if compra_show == 0:
        st.info("Compra recomendada: **0 unidades** (el stock actual cubre el objetivo definido).")
    else:
        st.warning(f"Compra recomendada: **{compra_show} unidades** (redondeado hacia arriba).")

    # 4) Gráfica: últimos 4 consumos + predicción semanal
    serie = pd.DataFrame({
        "Semana": ["-4", "-3", "-2", "-1", "Pred"],
        "Consumo": [cons_w4, cons_w3, cons_w2, cons_w1, pred_consumo]
    }).set_index("Semana")

    st.line_chart(serie)

    st.caption(
        "Nota: El modelo es un apoyo a la decisión. La recomendación depende de la política definida "
        f"({HORIZON_WEEKS} semanas + {int(SAFETY_FACTOR*100)}% seguridad) y de la calidad/continuidad del registro."
    )
