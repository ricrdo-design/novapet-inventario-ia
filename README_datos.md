# README Datos — NovaPet

## Propósito

Este documento describe el tratamiento aplicado a los datos utilizados en el proyecto **Sistema inteligente para la predicción del inventario de medicamentos y consumibles críticos en NovaPet mediante modelos de aprendizaje automático**.

Su objetivo es facilitar la revisión académica, proteger información sensible y permitir la reproducibilidad técnica del flujo de trabajo.

---

## Fuente de datos

La fuente original corresponde al Kardex operativo de NovaPet, utilizado con fines exclusivamente académicos para el desarrollo del proyecto de titulación.

El Kardex contiene información relacionada con movimientos de inventario, productos, categorías, consumos y registros operativos necesarios para construir el dataset predictivo.

---

## Datos incluidos en el repositorio

Por confidencialidad, el repositorio debe incluir únicamente una versión anonimizada, depurada o sustituta del dataset original.

Archivo recomendado:

```text
data/dataset_limpio_anonimizado.csv
```

También puede incluirse, si aplica:

```text
data/consumo_semanal_novapet.csv
```

---

## Variables conservadas

La versión anonimizada o sustituta conserva únicamente las variables necesarias para reproducir el flujo técnico del proyecto:

| Variable | Descripción |
|---|---|
| producto | Nombre del producto o identificador anonimizado |
| categoria | Tipo de producto: medicamento, vacuna, farmacia u otra categoría operativa |
| anio | Año calendario de referencia |
| semana | Semana calendario ISO |
| consumo_semanal | Consumo agregado semanal del producto |
| precio_unitario | Precio promedio unitario, si está disponible |
| lag_1 | Consumo de la semana anterior |
| lag_2 | Consumo de hace dos semanas |
| lag_3 | Consumo de hace tres semanas |
| lag_4 | Consumo de hace cuatro semanas |
| promedio_4 | Promedio móvil de las últimas cuatro semanas |
| min_4 | Consumo mínimo observado en las últimas cuatro semanas |
| max_4 | Consumo máximo observado en las últimas cuatro semanas |
| std_4 | Desviación estándar del consumo en las últimas cuatro semanas |
| tendencia_2 | Diferencia entre lag_1 y lag_2 |

---

## Datos excluidos o protegidos

Para proteger información sensible, no se incluyen:

- datos personales de clientes;
- datos personales de pacientes;
- información clínica individual;
- información financiera sensible;
- identificadores internos no necesarios;
- documentos operativos completos del Kardex original;
- registros que permitan reconstruir información confidencial de la organización.

---

## Anonimización aplicada

La anonimización o sustitución de datos puede incluir una o varias de las siguientes acciones:

1. Eliminación de campos no necesarios para el modelo.
2. Estandarización de nombres de productos.
3. Uso de identificadores genéricos cuando corresponda.
4. Conservación únicamente de consumos agregados semanales.
5. Eliminación de información personal o clínica.
6. Uso de dataset sintético o reducido para fines de demostración.

---

## Justificación metodológica

El objetivo del dataset incluido no es exponer información operativa sensible, sino permitir que el evaluador comprenda y reproduzca el flujo técnico del proyecto:

```text
Kardex → limpieza → variables temporales → modelos → validación → prototipo Streamlit
```

La versión anonimizada mantiene la estructura necesaria para ejecutar notebooks, revisar el pipeline y validar la lógica general del sistema.

---

## Limitaciones del dataset compartido

La versión incluida en el repositorio puede diferir del Kardex original en nivel de detalle o cantidad de registros, debido a criterios de confidencialidad.

Por esta razón, los resultados pueden variar si se ejecuta el flujo con un dataset reducido, anonimizado o sintético. Sin embargo, la estructura metodológica, el pipeline y la lógica de modelado se mantienen reproducibles.

---

## Uso permitido

Los datos incluidos en este repositorio deben utilizarse únicamente con fines académicos, de revisión metodológica y reproducibilidad del proyecto.

No deben utilizarse para fines comerciales ni para inferir información operativa sensible de NovaPet.

---

## Responsable del tratamiento académico

Proyecto desarrollado por:

- Juan Pablo López Cox
- Ricardo David Ayala Andrade

Maestría en Inteligencia Artificial Aplicada  
Universidad de Las Américas  
Mayo 2026
