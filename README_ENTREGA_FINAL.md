# Entrega Final - Proyecto NovaPet

## Sistema inteligente para la predicción del inventario de medicamentos y consumibles críticos en la clínica veterinaria NovaPet mediante modelos de aprendizaje automático

**Maestría en Inteligencia Artificial Aplicada**  
**Autores:** Juan Pablo López Cox y Ricardo David Ayala Andrade  
**Tutor:** Ana Estrella  
**Fecha:** 09/05/2026

---

## 1. Enlaces principales

> Reemplazar los campos entre corchetes con los enlaces finales antes de entregar.

| Recurso | Enlace | Propósito |
|---|---|---|
| Demo Streamlit | [PEGAR_ENLACE_STREAMLIT] | Ejecutar el prototipo funcional de predicción y recomendación de compra. |
| Repositorio GitHub | [PEGAR_ENLACE_GITHUB] | Revisar código fuente, dependencias, documentación y evidencias. |
| Informe final | `docs/informe_final.pdf` | Documento académico principal del proyecto Capstone. |
| Anexos de validación | `docs/anexos_validacion.pdf` | Evidencias técnicas de validación, robustez e interpretabilidad. |
| Manual de usuario | `docs/manual_usuario.pdf` o `manual_usuario.md` | Guía operativa para uso reproducible del prototipo. |
| Presentación final | `presentacion/pitch_final_novapet.pptx` | Presentación utilizada para la defensa oral del proyecto. |

---

## 2. Descripción breve del proyecto

El proyecto NovaPet desarrolla un prototipo funcional de inteligencia artificial para apoyar la predicción del consumo semanal de medicamentos, vacunas y consumibles críticos en una clínica veterinaria. La solución utiliza datos históricos del Kardex, modelos de aprendizaje automático y una aplicación desarrollada en Streamlit para transformar predicciones en recomendaciones operativas de compra.

El motor operativo principal del prototipo es un modelo baseline basado en promedio móvil de cuatro semanas, seleccionado por presentar el mejor desempeño global en la validación temporal. Random Forest se mantiene como referencia experimental secundaria por su valor comparativo e interpretativo mediante SHAP.

---

## 3. Archivos principales del repositorio

```text
novapet-inventario-ia/
│
├── app.py
├── requirements.txt
├── runtime.txt
├── README.md
├── README_ENTREGA_FINAL.md
├── manual_usuario.md
├── modelo_rf_inventario_novapet.pkl
│
├── docs/
│   ├── informe_final.pdf
│   ├── anexos_validacion.pdf
│   └── manual_usuario.pdf
│
├── resultados/
│   ├── metricas_finales.csv
│   ├── holdout_temporal.png
│   ├── walkforward_validation.png
│   ├── shap_importancia_global.png
│   └── shap_distribucion_impacto.png
│
├── capturas_streamlit/
│   ├── 01_validacion_tecnica.png
│   ├── 02_datos_entrada.png
│   ├── 03_resultado_operativo.png
│   ├── 04_comparacion_experimental.png
│   └── 05_visualizacion.png
│
├── notebooks/
│   ├── 01_preparacion_datos.ipynb
│   ├── 02_entrenamiento_modelos.ipynb
│   ├── 03_validacion_modelos.ipynb
│   └── 04_interpretabilidad_shap.ipynb
│
└── presentacion/
    └── pitch_final_novapet.pptx
```

---

## 4. Instrucciones de ejecución

### 4.1 Clonar repositorio

```bash
git clone [PEGAR_ENLACE_GITHUB]
cd novapet-inventario-ia
```

### 4.2 Instalar dependencias

```bash
pip install -r requirements.txt
```

### 4.3 Ejecutar prototipo

```bash
streamlit run app.py
```

### 4.4 Requisitos técnicos

```txt
streamlit==1.41.1
pandas==2.2.2
numpy==2.0.2
joblib==1.5.3
scikit-learn==1.6.1
matplotlib==3.9.0
```

Archivo recomendado para compatibilidad:

```txt
runtime.txt -> python-3.11
```

---

## 5. Pipeline general del proyecto

1. Extracción de datos históricos desde Kardex.
2. Limpieza, estandarización y consolidación semanal por producto.
3. Generación de variables temporales: lags, promedio móvil, tendencia, semana y año.
4. Entrenamiento y comparación de modelos: baseline, Random Forest y XGBoost.
5. Validación temporal: hold-out y walk-forward validation.
6. Interpretabilidad mediante SHAP sobre Random Forest.
7. Despliegue del prototipo en Streamlit.
8. Documentación, evidencias y presentación final.

---

## 6. Resultados finales del modelo

| Modelo | MAE | RMSE | R2 | Rol final |
|---|---:|---:|---:|---|
| Baseline - Promedio móvil 4 semanas | 1.797 | 3.067 | 0.583 | Motor principal del prototipo |
| Random Forest | 2.178 | 4.139 | 0.241 | Referencia experimental secundaria |
| XGBoost | 2.559 | 4.148 | 0.238 | Modelo exploratorio comparado |

**Conclusión técnica:** el baseline presentó el mejor desempeño global bajo validación temporal. Random Forest se mantiene como referencia experimental secundaria debido a su utilidad interpretativa mediante SHAP.

---

## 7. Evidencias disponibles

| Evidencia | Ubicación | Descripción |
|---|---|---|
| Métricas finales | `resultados/metricas_finales.csv` | Comparación MAE, RMSE y R2 de los modelos evaluados. |
| Hold-out temporal | `resultados/holdout_temporal.png` | Evidencia de división cronológica train-test. |
| Walk-forward validation | `resultados/walkforward_validation.png` | Validación robusta mediante ventanas temporales. |
| SHAP | `resultados/shap_importancia_global.png` | Interpretabilidad del modelo Random Forest. |
| Capturas Streamlit | `capturas_streamlit/` | Evidencia funcional del prototipo. |
| Anexos | `docs/anexos_validacion.pdf` | Validación, robustez e interpretación técnica. |

---

## 8. Prueba rápida sugerida para el tutor

1. Abrir la demo de Streamlit.
2. Seleccionar un producto del catálogo.
3. Confirmar que categoría y precio se autocompletan.
4. Seleccionar año y semana objetivo.
5. Ingresar consumos de las últimas cuatro semanas.
6. Ingresar stock actual.
7. Presionar **Calcular recomendación**.
8. Revisar consumo esperado, stock objetivo, compra sugerida, inversión estimada y riesgo de quiebre.
9. Activar comparación experimental con Random Forest.
10. Revisar visualización final.

---

## 9. Consideraciones de uso responsable

El prototipo debe utilizarse como herramienta de apoyo a la decisión. No reemplaza la supervisión humana ni automatiza compras de manera autónoma. La calidad de las recomendaciones depende de la consistencia del Kardex y de la actualización periódica de datos.

---

## 10. Estado final de entrega

| Componente | Estado |
|---|---|
| Prototipo Streamlit funcional | Completo |
| README técnico | Completo |
| Manual de usuario | Completo |
| Informe final | Completo |
| Anexos de validación | Completo |
| Presentación final | Completo |
| Evidencias y capturas | Completo |
| Repositorio GitHub organizado | Pendiente de verificación final |

