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

<hr>

## Optimización del modelo y decisión del negocio

### Estrategia de Modelado: Partición de Datos y Selección de Algoritmos

<div class="text-justify" markdown="1">

Una vez que las variables predictoras `Frequency`, `Monetary`, `Recency` y la variable a predecir `Churn` fueron consolidadas y depuradas de cualquier riesgo de fuga de datos, el conjunto estaba listo para la fase de entrenamiento.

Para garantizar que los modelos desarrollaran verdadera capacidad de generalización y no se limitaran a memorizar el historial (overfitting), se aplicó una división metodológica de los datos en una proporción **80/20**:
- **Conjunto de Entrenamiento (80%):** Utilizado exclusivamente para que los algoritmos matemáticos aprendieran los patrones de comportamiento que anteceden a la fuga de un cliente.
- **Conjunto de Prueba (20%):** Este subconjunto funciona como nuestro simulador de entorno de producción para validar la precisión real del modelo.

```python
from sklearn.model_selection import train_test_split

# Definimos las varibles predictivas (X) y la varible a predecir (Y)
X = rfm_log[['Frequency', 'Monetary', 'Recency']]
y = rfm_log['Churn']

# Dividimos los datos en 80% entrenamiento y 20% prueba 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size= 0.2, random_state= 42)
```

En lugar de forzar un único modelo, se diseñó una estrategia de evaluación comparativa entre dos enfoques de clasificación distintos:

La intención principal era utilizar **Random Forest Classifier** por su altísima capacidad predictiva. Random Forest es excelente para capturar relaciones no lineales complejas, requiere menos suposiciones sobre la distribución de los datos y tiene un riesgo bajo de sobreajuste. Este modelo se estableció como nuestra línea base de alto rendimiento.

 Aunque Random Forest es sumamente preciso, su naturaleza de "caja negra" dificulta explicar el porqué de una decisión a los equipos de venta o gerencia. Por ello, se entrenó en paralelo un modelo de **Regresión Logística**. El objetivo era evaluar si este modelo más tradicional lograba una precisión competitiva frente al Random Forest; de ser así, nos brindaría una ventaja fundamental en el mundo de los negocios. Tras entrenar y evaluar ambos modelos con el conjunto de prueba (el 20% de datos invisibles), obtuvimos los siguientes resultados:

| Modelo Predictivo | Precisión (Accuracy) |
| :--- | :--- |
| **Regresión Logística** | 0.9965 (99.65%) |
| **Random Forest Classifier** | 1.000 (100%) |

</div>

### Detección de fuga de datos y ajuste del modelo 

<div class="text-justify" markdown="1">

En la primera prueba de entrenamiento, incluyendo todas las características generadas en la matriz RFM, los modelos obtuvieron métricas de precisión extraordinariamente altas en el conjunto de prueba. 
En el mundo real de la ciencia de datos, un modelo con un rendimiento prácticamente perfecto casi nunca significa que el algoritmo sea un "éxito rotundo"; por el contrario, suele ser una clara señal de un error metodológico conocido como **Data Leakage (Fuga de Datos)**.

Al inspeccionar por qué los algoritmos alcanzaban la perfección, la causa fue evidente al revisar la definición de nuestra variable objetivo (`Churn`):

1. **La Regla de Negocio:** Se definió que un cliente está en estado de "Fuga" (`Churn = 1`) si su `Recency` es mayor a 90 días.
2. **El Conflicto en 'X':** Si mantenemos la variable `Recency` dentro del conjunto de variables predictoras (`X`), le estamos entregando al modelo la respuesta exacta del examen.

El algoritmo de **Random Forest** obtuvo 100% de precisión porque simplemente aprendió una regla condicional directa (Si Recency > 90 entonces Churn = 1), mientras que la **Regresión Logística** asignó un peso matemático casi absoluto a esa misma variable. El modelo no estaba aprendiendo a predecir hábitos o patrones de abandono; simplemente estaba recalculando la regla que nosotros mismos habíamos programado.

La solución para que un sistema predictivo útil para el negocio debe ser capaz de anticipar el riesgo de fuga analizando el comportamiento del cliente (cuánto gasta, qué tan seguido compra o qué variedad de productos consume), y no solo medir los días de inactividad cuando ya se cumplió el plazo.

Por esta razón, se tomó la decisión metodológica de **eliminar `Recency` de las variables predictoras (`X`)** y conservar únicamente las métricas de comportamiento. Se volvieron a entranar y evaluar con las nuevas variables y se obtuvo: 

```python
from sklearn.model_selection import train_test_split

# Definimos las varibles predictivas (X) y la varible a predecir (Y)
X = rfm_log[['Frequency', 'Monetary']]
y = rfm_log['Churn']

# Dividimos los datos en 80% entrenamiento y 20% prueba 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size= 0.2, random_state= 42)

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# 1. Inicializamos ambos modelos
modelo_lr = LogisticRegression(random_state=42)
modelo_rf = RandomForestClassifier(random_state=42)

# 2. Entrenamos (Ajustamos) los modelos con los datos de entrenamiento
modelo_lr.fit(X_train, y_train)
modelo_rf.fit(X_train, y_train)

# 3. Hacemos que ambos examinen los datos de prueba
predicciones_lr = modelo_lr.predict(X_test)
predicciones_rf = modelo_rf.predict(X_test)

# 4. Imprimimos el veredicto (Precisión)
print("--- Resultados del Duelo ---")
print(f"Precisión Regresión Logística: {accuracy_score(y_test, predicciones_lr):.4f}")
print(f"Precisión Random Forest:       {accuracy_score(y_test, predicciones_rf):.4f}")
```
| Modelo Predictivo | Precisión (Accuracy) |
| :--- | :--- |
| **Regresión Logística** | 0.7143 (71.43%) |
| **Random Forest Classifier** | 0.6429 (64.29%) |

**Regresión Logística le gano a Random Forest Classifier**. Esto debido a que Regresión Logistica trazó una línea matemáticamente limpia y funciono bastante bien. El algortimo de Random Forest al ser más complejo intento crear reglas más especificas para cada cliente en los datos de prueba, por lo tanto, nos quedamos con el algortimo de regresión Logística. Importante mencionar que <u>"un modelo más complejo no es mejor siempre, especialmente cuando tienes pocas características."</u>

</div>

### Matríz de Confusión y Reporte de Clasificación (Classification Report)

<div class="text-justify" markdown="1">

Tras obtener la precisión general de los modelos (71.43% para la Regresión Logística), surgió una pregunta analítica fundamental: ¿Es suficiente saber que el algoritmo acierta 7 de cada 10 veces? En proyectos de retención de clientes, la respuesta es no. 

Dado que nuestro conjunto de datos está naturalmente desbalanceado (hay más clientes activos que inactivos), un modelo podría obtener un porcentaje decente simplemente prediciendo que "nadie se va". Para entender el verdadero comportamiento y utilidad de las predicciones, fue estrictamente necesario implementar dos herramientas de evaluación más robustas:

**¿Por qué es necesaria la matriz de confusión?** 
La precisión global no nos indica qué tipo de equivocaciones está cometiendo el algoritmo. La Matriz de Confusión soluciona esto al desglosar las predicciones en cuatro escenarios vitales:
* **True Positives (TP):** Clientes que el modelo identificó correctamente en estado de fuga. (El acierto principal).
* **Falses Negatives (FN):** Clientes que realmente abandonaron la marca, pero que el modelo clasificó como "Activos". Para una empresa, este es el peor escenario, ya que no se emite ninguna alerta y se pierde la oportunidad de retenerlos.
* **Falses Positives (FP):** Clientes leales que el modelo etiquetó en riesgo por error. (El costo aquí es menor;por lo tanto, se les enviaría una promoción innecesaria).
* **True Negatives (TN):** Clientes activos identificados correctamente.

**¿Por qué se utilizó el reporte de clasifiación?**
Para cuantificar lo que observamos en la Matriz de Confusión, este reporte nos entrega tres métricas estadísticas clave que traducen los aciertos y errores en indicadores reales de rendimiento:
* **Recall:** De todos los clientes que realmente abandonaron, ¿qué porcentaje logró detectar nuestro modelo? Es decir, mide cuantos **TP** fueron detectados por el modelo.
* **Precisión (Precision):** De todos los clientes que el modelo etiquetó como "en riesgo de fuga", ¿cuántos realmente se fueron? 
* **F1-Score:** Representa el equilibrio perfecto entre el Recall y la Precisión. Es la métrica definitiva para evaluar la calidad del modelo cuando las categorías (Activos vs. Fuga) no son simétricas.
* **Support**: Es la cantidad de veces que aparece cada clase real en tus datos de prueba.

Se generó una matriz de confusión y un reporte de clasificación con el siguiente código: 

```python
from sklearn.metrics import confusion_matrix, classification_report

# Generamos la matriz de confusion a partir de la Regresión Logistica 
matriz_lr = confusion_matrix(y_test, predicciones_lr)
reporte_lr = classification_report(y_test, predicciones_lr)

print('Matriz de confusión:')
print(matriz_lr)
print('Reporte de clasificación')
print(reporte_lr)
```

| | Predicción del Modelo: Activo (0) | Predicción del Modelo: Fuga (1) |
| :--- | :--- | :--- |
| **Realidad: Cliente Activo (0)** | **461** (Verdaderos Negativos) | **100** (Falsos Positivos) |
| **Realidad: Cliente en Fuga (1)** | **148** (Falsos Negativos) | **159** (Verdaderos Positivos) |


| Categoría | Precisión (Precision) | Sensibilidad (Recall) | F1-Score | Soporte (Total de clientes) |
| :--- | :--- | :--- | :--- | :--- |
| **0 (Activos)** | 0.76 | 0.82 | 0.79 | 561 |
| **1 (Fuga)** | 0.61 | 0.52 | 0.56 | 307 |


</div>

### Optimización del modelo: Mejorar el Recall

<div class="text-justify" markdown="1">

Una precisión global de 71% establecia un buen punto de partida, el **Recall de 0.52** para la clase de Fuga (1) presentaba un área de oportunidad crítica. Este valor indicaba que el modelo solo lograba detectar al 52% de los clientes que realmente abandonaban la marca, dejando que casi la mitad de los desertores pasaran desapercibidos. El objetivo de esta nueva iteración fue someter los datos a una fase de ingeniería de características (Feature Engineering) más profunda para elevar esta métrica predictiva.

### Hipótesis: Introducción del Ticket Promedio

Recurriendo a métricas fundamentales del sector retail, se propuso integrar el **Ticket Promedio** (gasto promedio por transacción). La lógica de negocio sugería que una disminución en el volumen de compra por visita podría ser un síntoma temprano y poderoso de abandono.

Para implementar esto, se calculó la proporción utilizando los valores crudos (`Monetary` / `Frequency`) y, posteriormente, se aplicó la transformación logarítmica a esta nueva columna para mantener la consistencia con la estabilización y simetría del resto de los datos.

Se integró esta nueva característica al conjunto predictivo, se procedió a reentrenar la Regresión Logística y se generó un nuevo reporte de clasificación.

```python
# calcularemos el ticket promedio en los datos originales 
rfm_df['Ticket_Promedio'] = rfm_df['Monetary'] / rfm_df['Frequency']

# Lo aplicamos en el logarotimo y lo guardamos en nuestro data set 
rfm_log['Ticket_Promedio'] = np.log1p(rfm_df['Ticket_Promedio'])

# Redefinimos nuestras variable X
X_nueva = rfm_log[['Frequency', 'Monetary', 'Ticket_Promedio']]
y_nueva = rfm_log['Churn']

# Volvemos a dividir los datos para entrenar 
X_train_n, X_test_n, y_train_n, y_test_n = train_test_split(X_nueva, y_nueva, test_size= 0.2, random_state= 42)

# volvemos a entranar con Logistic Regression debido a que fue el modelo ganador
modelo_lr.fit(X_train_n, y_train_n)
predicciones_lr_n = modelo_lr.predict(X_test_n)

# Nuevo reporte de ticket promedio 
print('Reporte de ticket promedio:')
print(classification_report(y_test_n, predicciones_lr_n))
```

Contra las expectativas de optimizar la estdística del modelo, los resultados fueron idénticos. Las métricas, incluyendo el Recall crítico de 0.52, no mostraron absolutamente ninguna variación.

| Categoría | Precisión (Precision) | Sensibilidad (Recall) | F1-Score | Soporte (Total de clientes) |
| :--- | :--- | :--- | :--- | :--- |
| **0 (Activos)** | 0.76 | 0.82 | 0.79 | 561 |
| **1 (Fuga)** | 0.61 | 0.52 | 0.56 | 307 |

La respuesta nula a esta hipótesis de agregar la técnica de Ticket Promedio, se debe a que la respuesta no radicaba en el comportamiento del cliente, sino en las propiedades del álgebra y la naturaleza del algoritmo. Al calcular el Ticket Promedio y aplicar el logaritmo, matemáticamente se ejecutó la siguiente operación basándose en la ley de los logaritmos para cocientes, es decir: 

$$
\log\left(\frac{\text{Monetary}}{\text{Frequency}}\right) = \log(\text{Monetary}) - \log(\text{Frequency})
$$

La Regresión Logística es, por definición, un modelo lineal que asigna pesos (coeficientes) a las variables. Puesto que el conjunto de entrenamiento ya incluía `Log_Monetary` y `Log_Frequency` como variables independientes, el algoritmo ya estaba capturando implícitamente la relación lineal entre ambas.

Al introducir esta nueva columna, no se le proporcionó al modelo ninguna información nueva; simplemente se ingresó una redundancia. Este hallazgo demostró que para romper el techo del 0.52 de Recall, no bastaría con reciclar las mismas dimensiones matemáticas, sino que se requeriría información de una naturaleza distinta o un cambio en el umbral de decisión.

</div>

### Optimización del Umbral de Decisión: Priorizando la Retención

<div class="text-justify" markdown="1">

Por defecto, algoritmos como la Regresión Logística utilizan un umbral de decisión del 0.5 (50%). Esto significa que el modelo solo clasificará a un cliente como "en fuga" si está más de un 50% seguro de ello. 

Desde una perspectiva de negocio, esta configuración estándar resultó ser inútil para nuestro objetivo. Esperar a que el modelo tenga un 50% de certeza significa ignorar las señales tempranas de abandono, lo que explica por qué nuestro Recall estaba estancado en 0.52 (dejando escapar a casi la mitad de los desertores). En la industria del retail, es preferible pecar de precavidos: el costo de enviarle un descuento o correo de retención a un cliente leal por error (Falso Positivo) es mínimo en comparación con el impacto financiero de perder a un comprador real (Falso Negativo).

Para solucionar esto, era necesario modificar la sensibilidad del algoritmo. En lugar de aceptar el límite predeterminado, se desarrolló un ciclo iterativo (`for loop`) para evaluar el rendimiento del modelo a través de múltiples escenarios. 

El objetivo de este ciclo fue calcular la Precisión y el Recall para cada umbral de probabilidad entre el 10% y el 90% (0.1 a 0.9). Esto nos permitió visualizar el trade-off (compromiso) entre retener a la mayor cantidad de clientes posibles y no desperdiciar presupuesto en demasiados falsos positivos.

El ciclo for se ejecuto de la siguiente manera: 

```python
from sklearn.metrics import recall_score, precision_score

# Definimos una lista de umbrales a probar (del 10% al 90%, en saltos de 5%)
umbrales = np.arange(0.1, 0.95, 0.05)

# Lista vacía para guardar los resultados de cada ciclo
resultados_umbrales = []

for t in umbrales:
    # Clasificamos usando el umbral 't'
    pred_t = (probabilidades >= t).astype(int)
    
    # Calculamos las métricas clave para la clase 1 (Churn)
    recall = recall_score(y_test_v, pred_t, zero_division=0)
    precision = precision_score(y_test_v, pred_t, zero_division=0)
    
    # Guardamos los resultados
    resultados_umbrales.append({
        'Umbral': round(t, 2), 
        'Recall': round(recall, 3), 
        'Precision': round(precision, 3)
    })

# Convertimos la lista a un DataFrame para visualizarlo fácilmente
df_umbrales = pd.DataFrame(resultados_umbrales)

# Filtramos para ver solo las opciones donde salvamos al menos al 60% de los clientes
opciones_viables = df_umbrales[df_umbrales['Recall'] >= 0.60]
print("--- Opciones de Umbral para el Negocio ---")
print(opciones_viables)
```
A través del ciclo iterativo, generamos la siguiente tabla de sensibilidad, la cual nos permite visualizar cómo cambia la capacidad de retención (Recall) frente a la certeza de la alerta (Precisión):

| Umbral (Threshold) | Sensibilidad (Recall) | Precisión (Precision) |
| :---: | :---: | :---: |
| 0.10 | 0.987 | 0.427 |
| 0.15 | 0.971 | 0.469 |
| **0.20** | **0.912** | **0.501** |
| 0.25 | 0.863 | 0.527 |
| 0.30 | 0.795 | 0.547 |
| 0.35 | 0.723 | 0.572 |
| 0.40 | 0.638 | 0.596 |

## Conclusión Comercial: La Decisión del 20%

Al analizar los resultados arrojados por la iteración, se concluyó que el umbral más óptimo y rentable para el negocio es el de 0.20 (20%).

Con este nuevo límite de decisión, los resultados del modelo se transformaron radicalmente:

- **Recall (91%):** Logramos identificar y alertar sobre 279 de los 307 clientes totales que realmente estaban en proceso de fuga (Recall de 0.91).

- **Precisión (50%):** Obtuvimos una precisión de 0.50.

**¿Qué significa esto para la empresa?**

Al reducir el umbral al 20%, le estamos diciendo al modelo: <u>"Si ves incluso una ligera probabilidad (20%) de que un cliente se vaya, levanta la mano"</u>. Es cierto que ahora 1 de cada 2 alertas será una falsa alarma (clientes que no se iban a ir y aún así recibirán una campaña de retención). Sin embargo, a cambio de este margen de error, blindamos al negocio reteniendo al 91% de los verdaderos desertores, transformando un modelo predictivo estándar en una verdadera red de seguridad financiera.

</div>
