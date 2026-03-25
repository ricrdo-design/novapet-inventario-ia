# NovaPet - Predicción de Inventario con IA

## Descripción del proyecto

Este proyecto desarrolla un sistema inteligente para predecir el consumo semanal de medicamentos críticos en NovaPet, con el objetivo de optimizar la gestión de inventarios y apoyar la toma de decisiones de compra.

El sistema combina técnicas de análisis de series temporales y modelos de aprendizaje automático, evaluados bajo un enfoque comparativo riguroso.

---

## Objetivo

Desarrollar e implementar un modelo predictivo que estime el consumo semanal por ítem (unidades/semana), validando su desempeño mediante métricas cuantitativas y comparación frente a un modelo baseline.

---

## Enfoque metodológico

El pipeline del proyecto incluye:

- Transformación del Kardex en dataset supervisado
- Feature engineering (lags y promedio móvil)
- División temporal (train/test)
- Entrenamiento de modelos:
  - Baseline: promedio móvil de 4 semanas
  - Random Forest Regressor
  - XGBoost Regressor
- Evaluación con:
  - MAE
  - RMSE
  - R²

---

## Resultados clave

El modelo baseline (promedio móvil de 4 semanas) obtuvo el mejor desempeño:

- MAE: 1.797
- RMSE: 3.067
- R²: 0.583

Superando a los modelos Random Forest y XGBoost bajo las condiciones actuales del dataset.

---

## Aplicación (Streamlit)

Se desarrolló una aplicación interactiva que permite:

- Ingresar consumos históricos recientes
- Predecir el consumo semanal
- Calcular una recomendación de compra
- Comparar opcionalmente con XGBoost (modo experimental)

El sistema utiliza como motor principal el baseline, por ser el modelo con mejor validación.

---

## Estructura del repositorio
