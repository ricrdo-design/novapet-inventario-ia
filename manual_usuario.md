# Manual de Usuario  
## Sistema inteligente para la predicción de inventario — NovaPet

**Proyecto:** Sistema inteligente para la predicción del inventario de medicamentos y consumibles críticos en la clínica veterinaria NovaPet mediante modelos de aprendizaje automático.  
**Aplicación:** Prototipo funcional en Streamlit  
**Versión:** 1.0  
**Autores:** Juan Pablo López Cox y Ricardo David Ayala Andrade  
**Programa:** Maestría en Inteligencia Artificial Aplicada  
**Fecha:** Mayo 2026  

---

# 1. Propósito del manual

Este manual tiene como objetivo orientar al usuario final en el uso correcto del prototipo NovaPet para la predicción de inventario de medicamentos, vacunas y productos críticos.

El sistema permite estimar el consumo esperado de la próxima semana y generar una recomendación de compra basada en:

- consumo histórico reciente;
- stock actual disponible;
- stock de seguridad;
- precio unitario promedio;
- validación técnica del modelo predictivo.

El prototipo funciona como una herramienta de apoyo a la decisión. No reemplaza el criterio humano ni automatiza compras de forma definitiva.

---

# 2. Alcance del sistema

El prototipo está diseñado para apoyar la gestión de inventario en NovaPet mediante una interfaz sencilla y comprensible para usuarios no técnicos.

## 2.1 El sistema permite

- Seleccionar un producto desde un catálogo desplegable.
- Visualizar automáticamente su categoría y precio unitario promedio.
- Seleccionar el año y la semana a predecir.
- Visualizar el rango de fechas correspondiente a la semana proyectada.
- Ingresar consumos recientes de las últimas cuatro semanas.
- Ingresar el stock actual disponible.
- Definir un porcentaje de stock de seguridad.
- Calcular el consumo esperado de la próxima semana.
- Calcular la demanda proyectada a cuatro semanas.
- Calcular el stock objetivo.
- Generar una compra sugerida en unidades enteras.
- Estimar la inversión económica de la compra.
- Identificar riesgo de quiebre de stock.
- Comparar experimentalmente el resultado con Random Forest.
- Visualizar gráficamente el consumo reciente y las predicciones.

## 2.2 El sistema no permite

- Realizar compras automáticas.
- Conectarse directamente a un ERP.
- Actualizar el Kardex de forma automática.
- Reemplazar la aprobación humana.
- Predecir correctamente productos sin histórico suficiente.
- Garantizar exactitud absoluta en escenarios atípicos o no observados.

---

# 3. Enfoque de gestión por procesos

El uso del prototipo se organiza bajo una lógica de gestión por procesos, con el fin de facilitar su escalabilidad, trazabilidad y reproducibilidad.

## 3.1 Proceso general

```text
Entrada de datos → Procesamiento predictivo → Recomendación operativa → Revisión humana → Decisión de compra
```

## 3.2 SIPOC del proceso

| Elemento | Descripción |
|---|---|
| Supplier / Proveedor | Kardex histórico de NovaPet y usuario responsable del inventario |
| Input / Entrada | Producto, categoría, precio, año, semana, consumos recientes y stock actual |
| Process / Proceso | Cálculo de predicción, stock objetivo y compra sugerida |
| Output / Salida | Recomendación de compra, riesgo de quiebre e inversión estimada |
| Customer / Cliente | Responsable de compras, administración o gestión de inventario |

---

# 4. Roles recomendados

| Rol | Responsabilidad |
|---|---|
| Usuario operativo | Ingresar datos recientes de consumo y stock actual |
| Responsable de inventario | Revisar recomendación y validar coherencia con el contexto real |
| Responsable de compras | Aprobar o ajustar la compra sugerida |
| Administrador técnico | Mantener actualizados archivos, catálogo y modelo experimental |

---

# 5. Requisitos para usar la aplicación

Para utilizar el prototipo se requiere:

- acceso al enlace de Streamlit;
- navegador web actualizado;
- conexión a internet;
- datos recientes de consumo por producto;
- stock actual disponible;
- criterio operativo para validar la recomendación.

No se requiere conocimiento técnico en programación para utilizar la interfaz.

---

# 6. Acceso al sistema

El sistema se ejecuta desde una aplicación web desarrollada en Streamlit.

## 6.1 Abrir la aplicación

Ingresar al enlace de Streamlit proporcionado en el repositorio o documento de entrega final.

```text
[PEGAR AQUÍ LINK DE STREAMLIT]
```

## 6.2 Verificar carga de la interfaz

Al abrir la aplicación, el usuario debe visualizar el encabezado:

```text
NovaPet | Predicción de inventario
```

y las secciones principales:

1. Validación técnica del proyecto.
2. Datos de entrada.
3. Predicción y recomendación.
4. Comparación experimental.
5. Visualización.

---

# 7. Uso paso a paso del sistema

## 7.1 Seleccionar producto

En la sección **Producto a evaluar**, seleccionar el medicamento, vacuna o producto crítico desde la lista desplegable.

Ejemplo:

```text
MELOXICAM 1.5MG 10ML
```

El sistema completará automáticamente:

- categoría;
- precio unitario promedio.

Esto evita errores de digitación y mantiene consistencia en los datos de entrada.

## 7.2 Revisar categoría y precio

Después de seleccionar el producto, la aplicación mostrará automáticamente:

```text
Categoría: FARMACIA
Precio unitario promedio: 1.70 USD
```

Estos campos aparecen bloqueados para evitar modificaciones accidentales.

## 7.3 Seleccionar año y semana a predecir

En la sección **Semana proyectada**, ingresar:

- año;
- semana a predecir.

Ejemplo:

```text
Año: 2026
Semana a predecir: 19
```

El sistema mostrará automáticamente el rango de fechas correspondiente.

Ejemplo:

```text
Semana proyectada: lunes 4 de mayo de 2026 – domingo 10 de mayo de 2026
```

Esto permite que el usuario entienda claramente el periodo que se está proyectando.

## 7.4 Ingresar consumos recientes

En la sección **Consumos recientes**, ingresar el consumo real de las últimas cuatro semanas cerradas.

| Campo | Significado |
|---|---|
| Semana -4 | Consumo de hace cuatro semanas |
| Semana -3 | Consumo de hace tres semanas |
| Semana -2 | Consumo de hace dos semanas |
| Semana -1 | Consumo de la última semana cerrada |

Ejemplo:

```text
Semana -4: 4 unidades
Semana -3: 3 unidades
Semana -2: 3 unidades
Semana -1: 6 unidades
```

Los valores deben ingresarse en unidades enteras.

## 7.5 Ingresar stock actual

Ingresar el stock disponible actualmente para el producto seleccionado.

Ejemplo:

```text
Stock actual disponible: 4 unidades
```

Este dato es necesario para calcular la compra recomendada.

## 7.6 Definir stock de seguridad

Seleccionar el porcentaje de stock de seguridad mediante el control deslizante.

Valor recomendado por defecto:

```text
20%
```

El stock de seguridad permite cubrir variaciones razonables de demanda y reducir riesgo de quiebre.

## 7.7 Calcular recomendación

Presionar el botón:

```text
Calcular recomendación
```

El sistema generará:

- consumo esperado de la próxima semana;
- demanda proyectada a cuatro semanas;
- stock objetivo;
- compra sugerida;
- riesgo de quiebre;
- inversión estimada;
- comparación experimental con Random Forest;
- gráfico de consumo reciente y predicciones.

---

# 8. Interpretación de resultados

## 8.1 Consumo próxima semana

Representa la estimación de consumo para la siguiente semana.

Ejemplo:

```text
Consumo próxima semana: 4 unidades
```

El valor se muestra en unidades enteras porque no es operativo comprar o consumir fracciones de productos.

## 8.2 Demanda proyectada a cuatro semanas

Corresponde a la proyección del consumo esperado para cuatro semanas.

Ejemplo:

```text
Demanda 4 semanas: 16 unidades
```

Este horizonte se utiliza para orientar una recomendación de compra práctica.

## 8.3 Stock objetivo

Es la suma de la demanda proyectada más el stock de seguridad.

Ejemplo:

```text
Demanda 4 semanas: 16 unidades
Stock de seguridad: 4 unidades
Stock objetivo: 20 unidades
```

## 8.4 Compra sugerida

La compra sugerida se calcula como:

```text
Stock objetivo - Stock actual
```

Si el resultado es menor o igual a cero, el sistema recomienda no comprar.

Ejemplo:

```text
Stock objetivo: 20 unidades
Stock actual: 4 unidades
Compra sugerida: 16 unidades
```

## 8.5 Riesgo de quiebre

El sistema muestra un mensaje de riesgo según la relación entre stock actual y consumo esperado.

Ejemplo de riesgo alto:

```text
Riesgo alto: el stock actual no cubre el consumo esperado de la próxima semana.
```

Ejemplo de stock suficiente:

```text
Stock suficiente para cubrir el consumo esperado de la próxima semana.
```

## 8.6 Inversión estimada

El sistema calcula el costo aproximado de la compra sugerida.

Ejemplo:

```text
Compra sugerida: 16 unidades
Precio unitario: 1.70 USD
Inversión estimada: 27.20 USD
```

Este valor apoya la toma de decisiones financieras y operativas.

---

# 9. Comparación experimental con Random Forest

El sistema incluye una comparación experimental con Random Forest.

Esta comparación se mantiene con fines metodológicos y académicos, pero la decisión operativa principal se basa en el modelo baseline, debido a que obtuvo mejor desempeño global en la validación técnica.

## 9.1 Interpretación

- **Baseline:** motor principal del prototipo.
- **Random Forest:** referencia experimental secundaria.
- **XGBoost:** modelo exploratorio evaluado durante el proyecto.

El usuario puede activar o desactivar la comparación experimental desde la casilla correspondiente.

---

# 10. Visualización

Después de calcular la recomendación, el sistema muestra un gráfico con:

- consumos recientes;
- predicción baseline;
- predicción Random Forest, si está activada.

Este gráfico permite revisar visualmente la tendencia del consumo y comparar la predicción con el comportamiento reciente.

---

# 11. Reglas operativas de uso

Para asegurar un uso adecuado del sistema, se recomienda:

1. Verificar que los consumos ingresados correspondan a semanas cerradas.
2. Confirmar que el stock actual esté actualizado.
3. Revisar la recomendación antes de aprobar una compra.
4. No usar el sistema como única fuente de decisión.
5. Validar casos atípicos con criterio humano.
6. Actualizar el catálogo de productos si cambian precios o categorías.
7. Revisar periódicamente el desempeño del modelo.

---

# 12. Errores frecuentes y solución

| Situación | Posible causa | Acción recomendada |
|---|---|---|
| El resultado parece muy alto | Consumo reciente elevado o stock bajo | Revisar datos ingresados y validar con Kardex |
| No aparece riesgo alto | El stock cubre el consumo esperado | No requiere acción correctiva |
| No se ejecuta Random Forest | Falta archivo `.pkl` o error de compatibilidad | Verificar archivo `modelo_rf_inventario_novapet.pkl` |
| La semana no coincide con fechas esperadas | Uso de calendario ISO | Confirmar semana calendario ISO |
| Precio no corresponde | Catálogo desactualizado | Actualizar catálogo en `app.py` |
| El sistema recomienda 0 unidades | Stock actual suficiente | Validar si existen eventos futuros no considerados |

---

# 13. Consideraciones éticas

El sistema fue diseñado como herramienta de apoyo a la decisión. Su uso debe mantener supervisión humana.

## 13.1 Riesgos identificados

- dependencia excesiva del modelo;
- errores derivados de datos históricos incorrectos;
- sobreconfianza en recomendaciones automáticas;
- falta de actualización del Kardex;
- ausencia de variables externas como campañas o estacionalidad.

## 13.2 Mitigaciones recomendadas

- revisión humana antes de aprobar compras;
- auditoría periódica del Kardex;
- actualización de precios y catálogo;
- monitoreo mensual del desempeño;
- documentación de excepciones;
- recalibración futura con nuevos datos.

---

# 14. Buenas prácticas de escalabilidad

Para escalar el sistema en futuras versiones, se recomienda:

- ampliar el histórico transaccional del Kardex;
- incorporar variables externas como número de consultas, campañas, promociones y feriados;
- conectar el sistema a una base de datos;
- automatizar la actualización de consumos;
- implementar monitoreo de desempeño del modelo;
- documentar cambios en GitHub;
- versionar modelos entrenados;
- validar el sistema con usuarios operativos.

---

# 15. Control del proceso

Se recomienda revisar el proceso bajo un ciclo de mejora continua PDCA.

## 15.1 Plan

Definir productos críticos, horizonte de compra y responsables.

## 15.2 Do

Ingresar datos en el sistema y generar recomendación.

## 15.3 Check

Comparar recomendación con consumo real posterior.

## 15.4 Act

Ajustar parámetros, catálogo o modelo según resultados observados.

---

# 16. Reproducibilidad técnica

Para ejecutar el prototipo localmente:

## 16.1 Clonar repositorio

```bash
git clone https://github.com/TU-USUARIO/novapet-inventario-ia.git
```

## 16.2 Ingresar al proyecto

```bash
cd novapet-inventario-ia
```

## 16.3 Instalar dependencias

```bash
pip install -r requirements.txt
```

## 16.4 Ejecutar aplicación

```bash
streamlit run app.py
```

---

# 17. Archivos principales del proyecto

| Archivo | Descripción |
|---|---|
| `app.py` | Código principal de la aplicación Streamlit |
| `requirements.txt` | Dependencias necesarias |
| `runtime.txt` | Versión de Python usada en despliegue |
| `modelo_rf_inventario_novapet.pkl` | Modelo Random Forest experimental |
| `README.md` | Documentación técnica del proyecto |
| `manual_usuario.md` | Manual de usuario |
| `docs/` | Informes, anexos y documentos finales |
| `resultados/` | Métricas, gráficos y evidencias |
| `capturas_streamlit/` | Capturas funcionales del prototipo |
| `presentacion/` | Presentación final del pitch |

---

# 18. Cierre

El prototipo NovaPet permite transformar datos históricos de inventario en una recomendación operativa clara, trazable y entendible para usuarios no técnicos.

Su principal valor consiste en apoyar la toma de decisiones de abastecimiento mediante predicción semanal, cálculo de stock objetivo, estimación de compra sugerida y visualización de riesgo.

El sistema debe utilizarse siempre como apoyo a la decisión y mantenerse bajo supervisión humana, especialmente en productos críticos o escenarios atípicos.

---
