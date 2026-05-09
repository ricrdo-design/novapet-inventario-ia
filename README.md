# NovaPet | Predicción Inteligente de Inventario Veterinario

## Descripción del proyecto

Este proyecto desarrolla un prototipo funcional de inteligencia artificial para apoyar la predicción de demanda semanal de medicamentos y vacunas en NovaPet, una clínica veterinaria.

El objetivo principal es estimar el consumo esperado de productos críticos y generar recomendaciones de compra que reduzcan el riesgo de quiebres de stock, compras urgentes y sobreinventario.

El proyecto integra técnicas de analítica predictiva, validación temporal, interpretabilidad y visualización interactiva mediante Streamlit.

---

# Problema organizacional

NovaPet enfrentaba dificultades para anticipar el consumo de medicamentos y vacunas debido a la variabilidad operativa y a la ausencia de herramientas predictivas estructuradas.

Esto generaba riesgos como:

- quiebres de stock,
- compras urgentes,
- sobreinventario,
- capital inmovilizado,
- y dependencia de decisiones manuales.

El proyecto busca apoyar la toma de decisiones mediante un sistema predictivo interpretable y reproducible.

---

# Objetivo del proyecto

Desarrollar un prototipo funcional de inteligencia artificial capaz de:

- predecir consumo semanal,
- recomendar compras de inventario,
- reducir riesgo operativo,
- y mejorar la planificación de abastecimiento.

---

# Tecnologías utilizadas

- Python 3.11
- Streamlit
- Pandas
- NumPy
- Scikit-learn
- Matplotlib
- Joblib

---

# Estructura del proyecto

```text
novapet-inventario-ia/
│
├── app.py
├── requirements.txt
├── README.md
├── manual_usuario.md
├── modelo_rf_inventario_novapet.pkl
│
├── data/
│   ├── dataset_original.csv
│   └── dataset_limpio.csv
│
├── notebooks/
│   ├── entrenamiento_modelo.ipynb
│   ├── validacion_walkforward.ipynb
│   └── shap_analysis.ipynb
│
├── resultados/
│   ├── metricas_finales.csv
│   ├── walkforward.png
│   ├── holdout_temporal.png
│   ├── shap_summary.png
│   └── capturas_streamlit/
│
├── anexos/
│   ├── anexos_validacion.pdf
│   └── informe_final.pdf
│
└── presentacion/
    └── pitch_final.pptx
```

---

# Instalación y ejecución

## 1. Clonar repositorio

```bash
git clone https://github.com/TU-USUARIO/novapet-inventario-ia.git
```

---

## 2. Ingresar al proyecto

```bash
cd novapet-inventario-ia
```

---

## 3. Instalar dependencias

```bash
pip install -r requirements.txt
```

---

## 4. Ejecutar Streamlit

```bash
streamlit run app.py
```

---

# Requisitos técnicos

El proyecto fue desarrollado utilizando:

```txt
streamlit==1.41.1
pandas==2.2.2
numpy==2.0.2
joblib==1.5.3
scikit-learn==1.6.1
matplotlib==3.9.0
```

---

# Pipeline general del proyecto

El flujo metodológico implementado fue:

1. Extracción y limpieza de datos del Kardex
2. Generación de variables temporales
3. Construcción de baseline (promedio móvil)
4. Entrenamiento Random Forest y XGBoost
5. Validación hold-out temporal
6. Walk-forward validation
7. Evaluación comparativa de métricas
8. Interpretabilidad mediante SHAP
9. Desarrollo del prototipo Streamlit
10. Evaluación UX/UI y mejoras operativas

---

# Modelos evaluados

## Baseline (Promedio móvil 4 semanas)

Modelo principal del prototipo debido a su mejor desempeño en validación temporal.

## Random Forest

Utilizado como referencia experimental secundaria para comparación metodológica.

## XGBoost

Evaluado experimentalmente, aunque presentó menor desempeño que Random Forest y Baseline.

---

# Resultados principales

| Modelo | MAE | RMSE | R² |
|---|---|---|---|
| Baseline | 1.797 | 3.067 | 0.583 |
| Random Forest | 2.178 | 4.139 | 0.241 |
| XGBoost | 2.559 | 4.148 | 0.238 |

---

# Hallazgos relevantes

- El Baseline superó a los modelos complejos en validación temporal.
- Random Forest presentó mejor desempeño que XGBoost.
- El comportamiento relativamente estable del dataset favoreció modelos simples y robustos.
- La validación temporal fue crítica para evitar fuga de información.
- SHAP permitió interpretar la relevancia de variables temporales como lag_1 y lag_2.

---

# Características del prototipo Streamlit

El sistema permite:

- seleccionar productos desde catálogo desplegable,
- autocompletar categoría y precio,
- seleccionar semana objetivo,
- visualizar rango real de fechas,
- ingresar consumos recientes,
- estimar demanda semanal,
- calcular stock objetivo,
- recomendar compras,
- visualizar inversión estimada,
- detectar riesgo de quiebre,
- comparar experimentalmente Baseline vs Random Forest,
- y visualizar gráficamente las predicciones.

---

# Evidencias y anexos

El repositorio incluye:

- métricas finales,
- gráficas de validación,
- análisis SHAP,
- capturas del prototipo,
- anexos técnicos,
- informe final,
- y presentación del proyecto.

---

# Consideraciones éticas

El proyecto reconoce riesgos asociados a:

- dependencia excesiva del sistema,
- errores derivados de datos incompletos,
- posibles sesgos operativos,
- y sobreconfianza en automatización.

Por ello:

- el sistema funciona como apoyo a la decisión,
- no reemplaza supervisión humana,
- y requiere validación operativa continua.

---

# Limitaciones actuales

- histórico temporal todavía moderado,
- ausencia de variables externas,
- dependencia de calidad del Kardex,
- enfoque inicial sobre medicamentos críticos.

---

# Trabajo futuro

Se plantea:

- ampliar histórico,
- incorporar variables externas,
- automatizar reentrenamiento,
- integrar monitoreo continuo,
- y conectar el sistema con herramientas operativas reales.

---

# Autores

Proyecto desarrollado como trabajo de titulación para la Maestría en Inteligencia Artificial Aplicada.

Universidad de Las Américas (UDLA)

2026
