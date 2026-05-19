# README de Revisión Docente — Entrega S11

## Proyecto

**Sistema inteligente para la predicción del inventario de medicamentos y consumibles críticos en NovaPet mediante modelos de aprendizaje automático**

**Autores:** Juan Pablo López Cox y Ricardo David Ayala Andrade  
**Tutor:** Ana Estrella  
**Programa:** Maestría en Inteligencia Artificial Aplicada  
**Versión:** `v1.0-final`  
**Fecha:** Mayo 2026  

---

## 1. ¿Qué es este prototipo?

Este prototipo es una aplicación web desarrollada en Streamlit para apoyar la gestión de inventario de NovaPet.

Permite estimar el consumo esperado de medicamentos, vacunas y productos críticos, calcular stock objetivo y generar una recomendación de compra basada en datos históricos del Kardex.

El sistema utiliza como motor principal un baseline de promedio móvil de cuatro semanas, debido a que obtuvo el mejor desempeño en la validación temporal del proyecto.

Random Forest se mantiene como referencia experimental secundaria para comparación metodológica e interpretabilidad.

---

## 2. ¿Qué necesita el evaluador?

Para revisar el proyecto, el evaluador puede utilizar dos rutas:

## Ruta A — Prueba rápida vía Streamlit

Requiere:

- navegador web;
- enlace de la aplicación Streamlit;
- conexión a internet.

No requiere instalación local.

## Ruta B — Ejecución local

Requiere:

- Python 3.11;
- clonar o descargar el repositorio;
- instalar dependencias desde `requirements.txt`.

---

## 3. Enlaces principales

Completar antes de entregar:

```text
Demo Streamlit:
[PEGAR AQUÍ LINK DE STREAMLIT]

Repositorio GitHub:
[PEGAR AQUÍ LINK DE GITHUB]

Informe final:
docs/informe_final.pdf

Anexos de validación:
docs/anexos_validacion.pdf

Manual de usuario:
manual_usuario.md

Presentación:
presentacion/
```

---

## 4. Prueba rápida del docente en menos de 10 minutos

### Paso 1

Abrir la demo Streamlit.

```text
[PEGAR AQUÍ LINK DE STREAMLIT]
```

### Paso 2

Seleccionar el producto:

```text
MELOXICAM 1.5MG 10ML
```

### Paso 3

Usar los siguientes valores:

| Campo | Valor |
|---|---:|
| Año | 2026 |
| Semana a predecir | 19 |
| Semana -4 | 4 |
| Semana -3 | 3 |
| Semana -2 | 3 |
| Semana -1 | 6 |
| Stock actual | 4 |
| Stock de seguridad | 20% |

### Paso 4

Presionar:

```text
Calcular recomendación
```

### Paso 5

Resultado esperado:

El sistema debe mostrar:

- consumo esperado de la próxima semana;
- demanda proyectada a cuatro semanas;
- stock objetivo;
- compra sugerida en unidades enteras;
- riesgo de quiebre o stock suficiente;
- inversión estimada;
- comparación experimental con Random Forest;
- gráfico de consumo reciente y predicciones.

---

## 5. Ejecución local

### 5.1 Clonar repositorio

```bash
git clone [PEGAR AQUÍ LINK DE GITHUB]
```

### 5.2 Ingresar a la carpeta del proyecto

```bash
cd novapet-inventario-ia
```

### 5.3 Instalar dependencias

```bash
pip install -r requirements.txt
```

### 5.4 Ejecutar aplicación

```bash
streamlit run app.py
```

---

## 6. Estructura sugerida del repositorio

```text
novapet-inventario-ia/
│
├── README.md
├── README_ENTREGA_FINAL.md
├── README_REVISION_DOCENTE.md
├── README_datos.md
├── VERSION.md
├── CREDITOS_LICENCIAS.md
├── app.py
├── requirements.txt
├── runtime.txt
├── manual_usuario.md
├── modelo_rf_inventario_novapet.pkl
│
├── data/
├── notebooks/
├── resultados/
├── capturas_streamlit/
├── docs/
└── presentacion/
```

---

## 7. Artefactos principales

| Artefacto | Ubicación | Propósito |
|---|---|---|
| Aplicación Streamlit | `app.py` | Prototipo funcional |
| Modelo experimental | `modelo_rf_inventario_novapet.pkl` | Comparación Random Forest |
| README principal | `README.md` | Descripción técnica general |
| Manual de usuario | `manual_usuario.md` | Guía para usuario final |
| Notebooks | `notebooks/` | Flujo reproducible |
| Métricas | `resultados/metricas_finales.csv` | Resultados comparativos |
| Capturas | `capturas_streamlit/` | Evidencias de ejecución |
| Informe | `docs/informe_final.pdf` | Documento académico principal |
| Anexos | `docs/anexos_validacion.pdf` | Evidencias técnicas |

---

## 8. Resultados de referencia

| Modelo | MAE | RMSE | R² |
|---|---:|---:|---:|
| Baseline | 1.797 | 3.067 | 0.583 |
| Random Forest | 2.178 | 4.139 | 0.241 |
| XGBoost | 2.559 | 4.148 | 0.238 |

Interpretación:

El baseline de promedio móvil de cuatro semanas obtuvo el mejor desempeño general, por lo que se utiliza como motor principal del prototipo.

---

## 9. Notebooks reproducibles

La carpeta `notebooks/` contiene el flujo técnico por etapas:

| Notebook | Función |
|---|---|
| `01_preparacion_datos.ipynb` | Limpieza y construcción del dataset |
| `02_entrenamiento_modelos.ipynb` | Entrenamiento y comparación de modelos |
| `03_validacion_walkforward.ipynb` | Validación temporal y robustez |
| `04_interpretabilidad_shap.ipynb` | Interpretabilidad mediante SHAP |

---

## 10. Datos anonimizados

El repositorio debe incluir una versión anonimizada o sustituta del dataset en:

```text
data/dataset_limpio_anonimizado.csv
```

La documentación de tratamiento de datos se encuentra en:

```text
README_datos.md
```

---

## 11. Notas para el evaluador

- El prototipo es un MVP académico funcional.
- No automatiza compras.
- La decisión final debe mantenerse bajo supervisión humana.
- El desempeño puede variar si se ejecuta con un dataset reducido, anonimizado o sintético.
- El objetivo de la entrega es evidenciar funcionalidad, reproducibilidad y organización de artefactos.

---

## 12. Resultado esperado de la revisión

Al finalizar la revisión, el evaluador debería poder confirmar que:

- el repositorio está organizado;
- la app funciona;
- existe documentación técnica;
- existen notebooks reproducibles;
- existen evidencias de ejecución;
- los datos están anonimizados o protegidos;
- la versión final está identificada;
- las dependencias están declaradas;
- el flujo principal puede probarse en menos de 10 minutos.
