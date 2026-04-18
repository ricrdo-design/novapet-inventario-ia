# NovaPet | Predicción inteligente de inventario veterinario

Aplicación desarrollada en Streamlit para apoyar la toma de decisiones de compra de medicamentos y vacunas en la clínica veterinaria NovaPet.

## Objetivo

Estimar el consumo semanal esperado de productos críticos y recomendar compras futuras utilizando analítica predictiva.

---

# Lógica del sistema

## Modelo principal de operación

### Baseline (Promedio móvil de 4 semanas)

Se utiliza como motor principal porque obtuvo el mejor desempeño técnico durante la validación del proyecto:

| Modelo | MAE | RMSE | R² |
|-------|------|------|------|
| Baseline | 1.797 | 3.067 | 0.583 |
| Random Forest | 2.178 | 4.139 | 0.241 |
| XGBoost | 2.559 | 4.148 | 0.238 |

## Modelo experimental secundario

### Random Forest

Se mantiene dentro de la aplicación como benchmark comparativo para fines analíticos.

---

# Funcionalidades

- Predicción de consumo semanal
- Proyección para 4 semanas
- Stock de seguridad (+20%)
- Cálculo de stock objetivo
- Compra recomendada
- Comparación experimental con Random Forest
- Visualización gráfica

---

# Estructura del repositorio

```bash
app.py
modelo_rf_inventario_novapet.pkl
requirements.txt
README.md
