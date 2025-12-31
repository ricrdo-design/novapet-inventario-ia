import datetime as dt

import joblib
import numpy as np
import pandas as pd
import streamlit as st


st.set_page_config(page_title="NovaPet - Predicción de Inventario", layout="centered")
st.title("NovaPet | Predicción de inventario (Medicamentos y Vacunas)")
st.caption("Modelo: XGBoost | Alcance: productos con historial de salidas en Kardex.")


@st.cache_resource
def load_model(path: str):
    return joblib.load(path)

model = load_model("modelo_xgb_inventario.pkl")

# Extraer categorías válidas (evita inputs inválidos)
pre = model.named_steps["preprocess"]
ohe = pre.named_transformers_["cat"]
productos_validos = list(ohe.categories_[0])
categorias_validas = list(ohe.categories_[1])

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
# Límites razonables para evitar errores de captura (puedes ajustar)
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
anio = today.isocalendar().year
semana = today.isocalendar().week

# Features derivadas
lag_1 = float(cons_w1)
lag_2 = float(cons_w2)
promedio_4 = float(np.mean([cons_w1, cons_w2, cons_w3, cons_w4]))

# DataFrame para predicción
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

if st.button("Calcular recomendación"):
    pred_consumo = float(model.predict(X_pred)[0])

    # Estabilizar: no permitir predicción negativa por cualquier rareza del modelo
    pred_consumo = max(pred_consumo, 0.0)

    # Compra recomendada (no negativa)
    compra_recomendada = max(pred_consumo - float(stock_actual), 0.0)

    # Redondeos para operación
    pred_show = round(pred_consumo, 2)
    compra_show = int(np.ceil(compra_recomendada))

    st.success(f"Consumo esperado (próxima semana): **{pred_show} unidades**")
    st.info(f"Compra recomendada: **{compra_show} unidades** (redondeado hacia arriba)")

    # Gráfica: últimos 4 consumos + predicción
    serie = pd.DataFrame({
        "Semana": ["-4", "-3", "-2", "-1", "Pred"],
        "Consumo": [cons_w4, cons_w3, cons_w2, cons_w1, pred_consumo]
    }).set_index("Semana")

    st.line_chart(serie)
    st.caption("Nota: El modelo es un apoyo a la decisión. Su precisión depende de la calidad y continuidad del registro de consumos.")
