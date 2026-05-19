# VERSION

## Versión final del proyecto

**Nombre del proyecto:** Sistema inteligente para la predicción del inventario de medicamentos y consumibles críticos en NovaPet mediante modelos de aprendizaje automático.

**Versión:** `v1.0-final`  
**Fecha:** Mayo 2026  
**Estado:** Prototipo final consolidado para entrega S11 — Artefactos del Proyecto  
**Tipo de versión:** MVP funcional académico  

---

## Descripción de la versión

La versión `v1.0-final` consolida los principales artefactos técnicos, metodológicos y documentales del proyecto de titulación.

Incluye:

- prototipo funcional en Streamlit;
- modelo baseline como motor principal;
- Random Forest como referencia experimental secundaria;
- comparación con XGBoost;
- notebooks reproducibles;
- métricas finales;
- evidencias de validación temporal;
- análisis de interpretabilidad;
- documentación técnica;
- manual de usuario;
- presentación final;
- guía de entrega y revisión.

---

## Componentes principales incluidos

| Componente | Estado |
|---|---|
| `app.py` | Final |
| `requirements.txt` | Final |
| `runtime.txt` | Final |
| `README.md` | Final |
| `README_ENTREGA_FINAL.md` | Final |
| `manual_usuario.md` | Final |
| `modelo_rf_inventario_novapet.pkl` | Final |
| `notebooks/` | Final |
| `resultados/` | Final |
| `capturas_streamlit/` | Final |
| `docs/` | Final |
| `presentacion/` | Final |

---

## Modelo operativo final

El prototipo utiliza como motor principal:

```text
Baseline: promedio móvil de 4 semanas
```

Esta decisión se fundamenta en la validación temporal realizada, donde el baseline presentó el mejor desempeño global frente a Random Forest y XGBoost.

---

## Modelo experimental secundario

El proyecto mantiene como referencia experimental secundaria:

```text
Random Forest Regressor
```

Su función es permitir comparación metodológica e interpretabilidad mediante SHAP.

---

## Métricas finales de referencia

| Modelo | MAE | RMSE | R² |
|---|---:|---:|---:|
| Baseline | 1.797 | 3.067 | 0.583 |
| Random Forest | 2.178 | 4.139 | 0.241 |
| XGBoost | 2.559 | 4.148 | 0.238 |

---

## Cambios principales incorporados en la versión final

- Consolidación del prototipo en Streamlit.
- Mejora de interfaz UX/UI para usuario no técnico.
- Selección de producto mediante lista desplegable.
- Autocompletado de categoría y precio.
- Visualización de semana proyectada con rango de fechas.
- Resultados operativos en unidades enteras.
- Cálculo de stock objetivo y compra sugerida.
- Estimación de inversión económica.
- Mensajes de riesgo de quiebre.
- Comparación experimental con Random Forest.
- Documentación de reproducibilidad.
- Organización final del repositorio.

---

## Registro recomendado en GitHub

Se recomienda crear un release o tag con el siguiente nombre:

```text
v1.0-final
```

Título sugerido del release:

```text
NovaPet MVP Final — Entrega S11
```

Descripción sugerida:

```text
Versión final consolidada del prototipo NovaPet para la entrega S11 de Artefactos del Proyecto. Incluye aplicación Streamlit, modelo experimental, notebooks reproducibles, evidencias de validación, documentación técnica y presentación final.
```

---

## Nota de control de versiones

Se evita el uso de nombres ambiguos como:

```text
final_final
ultimo_final
version7
entrega_final_corregida
```

La versión oficial de revisión es:

```text
v1.0-final
```
