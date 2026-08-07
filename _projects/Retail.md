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

</div>

<hr>

## Descripción del dataset

<div class="text-justify" markdown="1">

Los datos analizados provienen de un registro transaccional crudo que contiene más de 500,000 operaciones históricas de comercio minorista (retail). En su estado original, el dataset registraba interacciones a nivel de factura, incluyendo identificadores de cliente, códigos de producto, cantidades y fechas.Puedes consultarlo en la siguiente dirección dando click [aquí](https://archive.ics.uci.edu/dataset/352/online+retail){:target="_blank" rel="noopener noreferrer"}

</div>

<hr>

## Problema del negocio 

<div class="text-justify" markdown="1">

Las empresas pierden miles de dólares anuales debido a la fuga de clientes (Churn). Como este abandono rara vez es explícito, los equipos de marketing suelen desperdiciar presupuesto enviando campañas de retención masivas, impactando negativamente el ROI al ofrecer descuentos a clientes que de igual manera iban a seguir comprando (Falsos Positivos), y dejando escapar a clientes valiosos por falta de detección oportuna (Falsos Negativos).

La Solución Propuesta: Desarrollar un modelo predictivo basado en el comportamiento histórico de compras (análisis RFM). El objetivo de este sistema es:
- Identificar con alta precisión a los clientes con mayor probabilidad de abandono.
- Optimizar el umbral de decisión matemática para priorizar la retención de clientes reales sobre el ruido estadístico.
- Transformar los datos crudos en una herramienta accionable que permita al departamento de finanzas y marketing justificar cada peso invertido en campañas de fidelización.

</div>

<hr>

## Procesamiento de datos e ingeniería de características (Feature Engineering)

<div class="text-justify" markdown="1">

En este proyecto, el conjunto original consistía en un registro transaccional crudo, por lo que fue necesario un procesamiento exhaustivo para traducir cada línea de compra en un perfil de comportamiento de cliente accionable para el algoritmo.

</div>

### 1. Limpieza de datos 

<div class="text-justify" markdown="1">

El primer paso consistio en depurar la base de datos para segurarse de que el modelo aprendiera exclusivamnete las transacciones validas que representaran ingresos reales para el negocio.
- Eliminación de registros incompletos: Se descartaron las filas que carecían de un identificador de cliente (Customer ID), ya que es imposible trazar el comportamiento de abandono sin saber a quién pertenece la compra; además solo se conservaron los valores mayores a cero (> 0) de las columnas **Quantity** y **UnitPrice**, debido a que necesitamos información óptima posteriormente para nuestro modelo. Para ello se ejecuto el siguiente código: 

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

Posteriormente de limpiar y filtrar nuestra base de datos, estos fueron los resultados:

| Data Frame antes de Purificar | Data Frame después de Purificar | 
| :----------- | :------------: | 
| 541 909, 8       |    397 884, 8    | 

</div>

<hr>

### 2. Modelo RFM (Recency, Frequency, Monetary)

<div class="text-justify" markdown="1">

Un algoritmo de Machine Learning no puede predecir el abandono a partir de una lista de tickets de compra aislados. En este caso, el conjunto de datos original estaba a nivel transaccional (cada fila representaba un producto dentro de una factura), lo cual imposibilita predecir el comportamiento individual del usuario. Para solucionar esto, se requería tranformar el historial de tickets de compra en perfiles de comportamiento únicos por cliente. Se optó por la metodología **RFM (Recency, Frequency, Monetary),** que es un estándar en la inteligencia de negocios que perfila a los clientes en tres dimensiones claves: 

- **Recency (Recencia):** La cantidad de días transcurridos desde la última compra del cliente. 
- **Frequency (Frecuencia):** El número total de compras realizadas por el usuario, lo que indica su nivel de lealtad e interacción.
- **Monetary (Monetario):** El gasto total acumulado a lo largo de su cilo de vida con la empresa 

Este modelo permite traducir millones de filas de ventas en tres indicadores clave que el algoritmo puede interpretar fácilmente para detectar cuándo un cliente está modificando sus hábitos de consumo y, por ende, en riesgo de fuga.

</div>

#### ¿Cómo se realizo?

<div class="text-justify" markdown="1">

El progreso de agregación se ejecuto en tres fases estratégicas utilizando la librería **pandas:**

1. **Calculo del valor real:** Primero, se generó una nueva característica calculando el gasto total por línea de producto, multiplicando la cantidad de artículos por su precio unitario.
2. **Deficnición del horizonte temporal:** Para medir cuánto tiempo había pasado desde la última compra de un cliente, se estableció una fecha de referencia, esto para evitar sesgos; se simulo que le día de hoy era exactamente un día después de la transacción más reciente registrada en todo el conjunto de datos.
3. **Agregación Multidimensional (Grouping):** Se agrupo toda la información utilizando el identificador único del cliente **(CustomerID)** y se aplicaron funciones de agregación específicas para cada pilar del RFM:
    - **Recency:** Se calculó la diferencia en días entre la fecha de referencia y la fecha de la última compra del cliente.
    - **Frequency:** Se contabilizó el número de facturas únicas **(InvoiceNo)** asociadas al cliente, midiendo su recurrencia. 
    - **Monetary:** Se sumó el total de las ventas generadas por el cliente durante todo su ciclo de vida registrado.

El código que se implemento para esta transformación fue el siguiente: 

```python
# Crear la columna (Asegurando que el nombre coincida exactamente)
df['Total_Sales'] = df['Quantity'] * df['UnitPrice']

# Definir la fecha de referencia (el "hoy" de nuestro dataset)
fecha_referencia = df['InvoiceDate'].max() + pd.to_timedelta(1, unit='D')

# Calcular las variables RFM por cliente en un solo paso
rfm_df = df.groupby('CustomerID').agg({
    'InvoiceDate': lambda x: (fecha_referencia - x.max()).days,  # Recencia
    'InvoiceNo': 'nunique',                                      # Frecuencia
    'Total_Sales': 'sum'                                         # Monetario
})

# Renombrar las columnas para que el DataFrame sea más fácil de leer
rfm_df.rename(columns={
    'InvoiceDate': 'Recency',
    'InvoiceNo': 'Frequency',
    'Total_Sales': 'Monetary'
}, inplace=True)

print(rfm_df.head())
```

| Recency | Frequency | Monetary |
|:---|:---|:---|
| 326 | 1 | 77183.6 |
| 2 | 7 | 4310 |
| 75 | 4 | 1797.24 |
| 19 | 1 | 1757.55 |
| 310 | 1 | 334.4 |

Al generar esta nueva matriz, obtuvimmos el perfil consodilado de **4338 clientes únicos**. La estadística descriptiva de esta nueva población revela comportamientos muy interesantes para el negocio: 

| Estadística | Recency | Frequency | Monetary |
| :--- | :--- | :--- | :--- |
| **count** | 4,338 | 4,338 | 4,338 |
| **mean** | 92.5 | 4.2 | 2,054.26 |
| **std** | 100.01 | 7.6 | 8,989.23 |
| **min** | 1.0 | 1.0 | 3.75 |
| **25%** | 18.0 | 1.0 | 307.41 |
| **50%** | 51.0 | 2.0 | 674.48 |
| **75%** | 142.0 | 5.0 | 1,661.74 |
| **max** | 374.0 | 209.0 | 280,206.02 |

Al observar la tabla, es evidente que esxiste una dispersión masiva en el gasto **(Monetary)** mientras que el clinete típico (la mediana) gasta $674.48; el valor máximo asciende a más de $280,000. Esta fuerte asimetría nos indicá inmediatamente que los datos crudos van a generar un sesgo a los modelos predictivos hacia los clientes, haciendo una transformación logaritmica antes de la fase de entrenamiento. 
</div>

### 3. Transformación Logarítmica: Tratamiento de Outliers y Asímetría

<div class="text-justify" markdown="1">

Como se destacó en la estadística descriptiva previa, variables como el Gasto Total (`Monetary`) y `Frequency` presentaban una asimetría positiva extrema. Mientras que el cliente promedio (la mediana) gastaba alrededor de $674, existían valores atípicos (outliers) o clientes ballena con gastos superiores a los $280,000.

Los algoritmos predictivos de clasificación, especialmente los paramétricos como la Regresión Logística, son altamente sensibles a estas magnitudes extremas. Si el modelo se hubiera entrenado con los datos crudos, el algoritmo habría sesgado sus pesos matemáticos para intentar ajustarse a ese pequeño grupo atípico, ignorando las sutilezas del comportamiento de abandono del cliente promedio, quien representa el verdadero volumen y sustento del negocio.

El objetivo de la transformación es normalizar la distribución matemáticamente para estas distancias extremas sin perder la jerarquía y varianza de los datos, es decir, el cliente que gasta más sigue teniendo el valor más alto, pero en una escala más densa. Esto permite que el modelo evalúe el riesgo de fuga de manera equitativa en todos los segmentos de clientes, mejorando su capacidad de generalización y asegurando que las alertas de retención se dirijan a clientes representativos.

Se aplicó una transfromación logarítmica utilizando la librería `Numpy`. Se utilizó la función `np.log1p`, debido para evitar errores computacionales con valores cero, es decir, que si hay algún cliente con valor 0, seguira siendo cero. Después de hacer la transformación logarítmica nos dio: 

| Estadística | Recency (Log) | Frequency (Log) | Monetary (Log) |
| :--- | :--- | :--- | :--- |
| **count** | 4,338 | 4,338 | 4,338 |
| **mean** | 3.831 | 1.346 | 6.594 |
| **std** | 1.340 | 0.683 | 1.258 |
| **min** | 0.693 | 0.693 | 1.558 |
| **25%** | 2.944 | 0.693 | 5.731 |
| **50%** | 3.951 | 1.099 | 6.515 |
| **75%** | 4.963 | 1.792 | 7.416 |
| **max** | 5.927 | 5.347 | 12.543 |

A continuación se muestra la diferencia de los datos crudos, antes y después de la transformación logarítmica. 

<div class="row justify-content-center">
  <div class="col-sm-12 mt-3 mt-md-4">
    {% include figure.liquid loading="eager" path="assets/img/normalvslog.jpg" title="Histogramas Crudo Vs Log" class="img-fluid rounded z-depth-1" %}
  </div>
</div>
<div class="caption text-center mt-2">
    Figura 1. Izquierda: Histograma con los datos crudos (Sesgada) | Derecha: Histograma posterior a la transformación logarítmica.
</div>


</div>

<hr>

## Definición de la Variable Objetivo: Estableciendo el Umbral de Abandono.

<div class="text-justify" markdown="1">

Para entrenar un modelo de Machine Learning supervisado, es indispensable contar con una variable objetivo que le indique al algoritmo qué es exactamente lo que estamos buscando. En este caso, necesitábamos definir una regla de negocio clara para clasificar a la base de usuarios en dos categorías: clientes Activos (0) o en estado Inactivo (1).

A esta variable objetivo la llamaremos `Churn`, que es el dictamen de días que sabemos si perdemos o aún tenemos un cliente.

A diferencia de un servicio de suscripción donde el usuario cancela explícitamente su contrato, en el sector retail el abandono es silencioso. Para establecer un punto de corte justo y basado en datos empíricos, se recurrió a la estadística descriptiva del análisis RFM. 

Dado que el promedio de Recencia (días desde la última compra) de toda la población de clientes era de aproximadamente 92.5 días, se determinó estratégicamente que **90 días de inactividad** es un rango de tiempo razonable y representativo para diagnosticar que un cliente ha interrumpido sus hábitos de consumo con la marca.

Con esta regla de negocio, se creó la columna `Churn` y se integró a la matriz de características:

```python
umbral_dias = 90 

# Creamos la columna Churn (1 si pasaron más días que el umbral, 0 si no)
rfm_df['Churn'] = (rfm_df['Recency'] > umbral_dias).astype(int)

# Y también se la agregamos a nuestro dataframe logarítmico para usarlo en el modelo
rfm_log['Churn'] = rfm_df['Churn']

# con value_counts observamos cuantos clientes hemos perdido y cuantos hemos retenido.
print('Clientes actualmente:')
print(rfm_df['Churn'].value_counts())
```
Al aplicar este umbral de 90 días a la base de datos, el algoritmo clasificó el estado actual de nuestro portafolio de la siguiente manera:

| Estado del Cliente | Etiqueta (Churn) | Cantidad de Clientes |
| :--- | :--- | :--- |
| **Activos** | `0` | 2,889 |
| **En Fuga (Inactivos)** | `1` | 1,449 |

Esta distribución nos confirma que tenemos un conjunto de datos desbalanceado (como es natural en los problemas de retención), con aproximadamente un 33% de la base de clientes clasificada como fuga.


</div>
