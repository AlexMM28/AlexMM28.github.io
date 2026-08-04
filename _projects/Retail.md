---
layout: page
title: "Reporte de Caso: Optimización de Retención de Clientes con predicción Churn y Maximización de ROI"
description: Alejandro Maldonado-Medina
img: assets/img/cover_retail.jpg
importance: 2
category: Ciencia de Datos
giscus_comments: true
pretty_table: true
---

## Resumen 

<div class="text-justify" markdown="1">

Desarrollo de un algoritmo predictivo para la prevención de Churn que asegura la retención del 91% de los clientes en riesgo. El proyecto destaca por la identificación temprana de sesgos (Data Leakage) y el ajuste de umbrales basados en reglas de negocio, logrando un equilibrio perfecto entre sensibilidad estadística y rentabilidad comercial.

## Descripción del dataset

<div class="text-justify" markdown="1">

Los datos analizados provienen de un registro transaccional crudo que contiene más de 500,000 operaciones históricas de comercio minorista (retail). En su estado original, el dataset registraba interacciones a nivel de factura, incluyendo identificadores de cliente, códigos de producto, cantidades y fechas.Puedes consultarlo en la siguiente dirección dando click [aquí](https://archive.ics.uci.edu/dataset/352/online+retail){:target="_blank" rel="noopener noreferrer"}

## Problema del negocio 

<div class="text-justify" markdown="1">

Las empresas pierden miles de dólares anuales debido a la fuga de clientes (Churn). Como este abandono rara vez es explícito, los equipos de marketing suelen desperdiciar presupuesto enviando campañas de retención masivas, impactando negativamente el ROI al ofrecer descuentos a clientes que de igual manera iban a seguir comprando (Falsos Positivos), y dejando escapar a clientes valiosos por falta de detección oportuna (Falsos Negativos).

La Solución Propuesta: Desarrollar un modelo predictivo basado en el comportamiento histórico de compras (análisis RFM). El objetivo de este sistema es:
- Identificar con alta precisión a los clientes con mayor probabilidad de abandono.
- Optimizar el umbral de decisión matemática para priorizar la retención de clientes reales sobre el ruido estadístico.
- Transformar los datos crudos en una herramienta accionable que permita al departamento de finanzas y marketing justificar cada peso invertido en campañas de fidelización.

## Procesamiento de datos e ingeniería de características (Feature Engineering)

<div class="text-justify" markdown="1">

En este proyecto, el conjunto original consistía en un registro transaccional crudo, por lo que fue necesario un procesamiento exhaustivo para traducir cada línea de compra en un perfil de comportamiento de cliente accionable para el algoritmo.

### 1. Limpieza de datos 

<div class="text-justify" markdown="1">

El primer paso consistio en depurar la base de datos para segurarse de que el modelo aprendiera exclusivamnete las transacciones validas que representaran ingresos reales para el negocio.
- Eliminación de registros incompletos: Se descartaron las filas que carecían de un identificador de cliente (Customer ID), ya que es imposible trazar el comportamiento de abandono sin saber a quién pertenece la compra. Se ejecuto el siguiente código: 

```python
#Visualización de datos nulos en el Data Frame 
sum_null = df.isnull().sum()
print('Datos Nulos:')
print(sum_null)

# Eliminar nulos de CustomerID
df = df.dropna(subset='CustomerID')

# Filtrar Quantity > 0
df = df[df['Quantity']>0]

# Filtrar UnitPrice > 0
df = df[df['UnitPrice']>0]

# Tamaño del Data Frame purificado
print(f'Tamaño del Data Frame actualizaco:{df.shape}')
```

| Left aligned | Center aligned | Right aligned |
| :----------- | :------------: | ------------: |
| Left 1       |    center 1    |       right 1 |
| Left 2       |    center 2    |       right 2 |
| Left 3       |    center 3    |       right 3 |

