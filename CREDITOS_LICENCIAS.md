# Créditos, Licencias y Atribuciones

## Proyecto

**Sistema inteligente para la predicción del inventario de medicamentos y consumibles críticos en NovaPet mediante modelos de aprendizaje automático**

Proyecto desarrollado como trabajo de titulación de la Maestría en Inteligencia Artificial Aplicada.

**Autores:**

- Juan Pablo López Cox
- Ricardo David Ayala Andrade

**Tutor / Guía:** Ana Estrella  
**Institución:** Universidad de Las Américas  
**Fecha:** Mayo 2026  

---

## Propósito de este documento

Este documento consolida las atribuciones correspondientes a herramientas, librerías, frameworks, recursos técnicos y datos utilizados durante el desarrollo del proyecto.

Su objetivo es cumplir con criterios de transparencia, trazabilidad académica y correcta atribución de recursos externos.

---

## Herramientas y tecnologías utilizadas

| Herramienta / Librería | Uso dentro del proyecto | Referencia |
|---|---|---|
| Python | Lenguaje principal de programación | Python Software Foundation |
| Streamlit | Desarrollo del prototipo web interactivo | Streamlit |
| Pandas | Manipulación y preparación de datos | The pandas development team |
| NumPy | Operaciones numéricas | NumPy Developers |
| Scikit-learn | Modelado Random Forest y métricas | Scikit-learn Developers |
| XGBoost | Modelo exploratorio de boosting | XGBoost Developers |
| Matplotlib | Visualización de resultados | Matplotlib Development Team |
| SHAP | Interpretabilidad de modelos | Lundberg & Lee |
| Joblib | Serialización del modelo `.pkl` | Joblib Developers |
| GitHub | Control de versiones y repositorio | GitHub |
| Streamlit Cloud | Despliegue del prototipo | Streamlit |

---

## Librerías principales declaradas en `requirements.txt`

```text
streamlit==1.41.1
pandas==2.2.2
numpy==2.0.2
joblib==1.5.3
scikit-learn==1.6.1
matplotlib==3.9.0
```

Si se ejecutan notebooks de interpretabilidad o entrenamiento extendido, pueden requerirse adicionalmente:

```text
xgboost
shap
openpyxl
```

---

## Datos utilizados

La fuente de datos original corresponde al Kardex operativo de NovaPet, utilizado con fines académicos para el desarrollo del proyecto.

Por criterios de confidencialidad, el repositorio debe incluir únicamente una versión anonimizada, reducida o sustituta del dataset, evitando exposición de información operativa sensible.

No se utilizan datos personales de clientes ni pacientes en el modelo predictivo final.

---

## Modelos evaluados

Durante el proyecto se evaluaron los siguientes enfoques:

- Baseline de promedio móvil de cuatro semanas.
- Random Forest Regressor.
- XGBoost Regressor.

El baseline fue seleccionado como motor principal del prototipo por presentar mejor desempeño en validación temporal.

Random Forest se mantuvo como referencia experimental secundaria debido a su utilidad para comparación e interpretabilidad.

---

## Referencias bibliográficas principales

Breiman, L. (2001). Random forests. *Machine Learning, 45*(1), 5–32. https://doi.org/10.1023/A:1010933404324

Chen, T., & Guestrin, C. (2016). XGBoost: A scalable tree boosting system. In *Proceedings of the 22nd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining* (pp. 785–794). ACM. https://doi.org/10.1145/2939672.2939785

Hyndman, R. J., & Athanasopoulos, G. (2021). *Forecasting: Principles and practice* (3rd ed.). OTexts. https://otexts.com/fpp3/

Lundberg, S. M., & Lee, S. I. (2017). A unified approach to interpreting model predictions. *Advances in Neural Information Processing Systems, 30*, 4765–4774.

Molnar, C. (2022). *Interpretable machine learning* (2nd ed.). Lulu.com.

Russell, S., & Norvig, P. (2021). *Artificial intelligence: A modern approach* (4th ed.). Pearson.

---

## Licencia del proyecto

Este repositorio se presenta con fines académicos.

Salvo que se indique lo contrario, el código desarrollado por los autores puede considerarse de uso académico y demostrativo.

Antes de reutilizarlo en entornos productivos, se recomienda:

- validar permisos sobre los datos;
- revisar seguridad y privacidad;
- realizar pruebas operativas adicionales;
- definir una licencia formal de software si el proyecto será publicado o reutilizado.

---

## Nota de responsabilidad

El prototipo desarrollado no debe interpretarse como un sistema autónomo de compra o abastecimiento. Su función es apoyar la toma de decisiones mediante predicciones y recomendaciones sujetas a revisión humana.
