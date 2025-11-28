---
layout: page
title: Predicción del costo de seguros médicos con Machine Learning.
description: Alejandro Maldonado-Medina
img: assets/img/insurance_cover.jpg
importance: 2
category: Ciencia de Datos
giscus_comments: true
---

## DATASET

<div class="text-justify" markdown="1">

El dataset utilizado en este proyecto proviene de la plataforma Kaggle y contiene información sobre características demográficas, hábitos y factores médicos relacionados con el costo de seguros de salud. Puedes consultarlo en la siguiente dirección dando click [aquí](https://www.kaggle.com/datasets/mosapabdelghany/medical-insurance-cost-dataset){:target="_blank" rel="noopener noreferrer"}

</div>

<hr>

## Descripción del proyecto

<div class="text-justify" markdown="1">

Este proyecto consiste en analizar y modelar el costo de los seguros médicos utilizando técnicas de Ciencia de Datos y Machine Learning. A partir del dataset **Insurance**, que incluye información como edad, índice de masa corporal (BMI), número de hijos, tabaquismo, sexo y región, se desarrolla un proceso completo que abarca desde el análisis exploratorio de datos (EDA) hasta la implementación de modelos predictivos.

El objetivo principal es **predecir el costo del seguro médico** con base en características demográficas y de salud de los asegurados. Para lograrlo, se aplican distintas etapas de limpieza, transformación de datos, visualización y prueba de varios modelos de regresión supervisada.

Los objetivos específicos del proyecto son:

- Comprender la estructura del dataset y analizar las relaciones entre variables por medio de visualizaciones estadísticas.
- Aplicar una adecuada preparación de datos mediante codificación de variables categóricas, estandarización y división de datos en entrenamiento y prueba.
- Construir distintos modelos de Machine Learning (Regresión Lineal, Árbol de Decisión y K-Nearest Neighbors) para analizar y comparar su desempeño predictivo.
- Evaluar los modelos utilizando métricas como R² y MAE.
- Identificar las variables que más influyen en la determinación del costo del seguro.

Este proyecto forma parte del portafolio de Ciencia de Datos y tiene como finalidad demostrar habilidades en análisis exploratorio, preprocesamiento de datos, modelado predictivo y comunicación de resultados.

</div>

<hr>

## Metodología 

<div class="text-justify" markdown="1">

La metodología aplicada en este proyecto sigue un flujo de trabajo típico de Ciencia de Datos, comenzando con la exploración del conjuntos de datos, seguido de su preparación para el modelado. El proceso se divide en dos etapas principales: Análisis Exploratorio de Datos y Procesamiento/Preprocesamiento de los datos.

### 1. Análisis Exploratorio de Datos 

Se realizó un análisis exploratorio completo con el objetivo de comprender la estructura del dataset y detectar patrones relevantes en las variables.
Las actividades principales fueron: 
- Revisión inicial del dataset: dimensiones, tipos de datos, valores faltantes y estadísticas descriptivas.

```python
df = pd.read_csv("insurance.csv")
print(df.head())
print(df.info())
print(df.describe())
print(df.isnull().sum())
```
- Visualización de distribuciones: histogramas y boxplots para identificar la forma de las distribuciones y posibles valores atípicos. 

<div class="row justify-content-center">
  <div class="col-sm-8 mt-3 mt-md-0">
    {% include figure.liquid 
      loading="lazy" 
      path="assets/img/histograms1.jpg" 
      title="histograma" 
      class="img-fluid rounded z-depth-1"
    %}
  </div>

  <div class="col-sm-8 mt-3 mt-md-4">
    {% include figure.liquid 
      loading="lazy" 
      path="assets/img/boxplots1.jpg" 
      title="Boxplot" 
      class="img-fluid rounded z-depth-1"
    %}
  </div>
</div>

<div class="caption text-center mt-2">
    Figura 1. Arriba: Histogramas de distribuciones de las varibles: Age, BMI, Children y Charges  | Abajo: Boxplots de las variables: Age, BMI, Children y Charges.
</div>

- Análisis de correlación: se generó un mapa de calor (heatmap) paara conocer las relaciones entre variables númericas, destacando la fuerte relación entre el tabaquismo y el costo del seguro. 

<div class="row justify-content-center">
  <div class="col-sm-8 mt-3 mt-md-4">
    {% include figure.liquid loading="eager" path="assets/img/hmap.jpg" title="Mapa de calor" class="img-fluid rounded z-depth-1" %}
  </div>
</div>
<div class="caption text-center mt-2">
    Figura 2. Análisis de correlación representado en un mapa de calor (Heatmap)
</div>

- Comparaciones entre grupos: se realizaron gráficos para analizar diferencias en cargos médicos entre fumadores y no fumadores, así como entre regiones. 

<div class="row justify-content-center">
  <div class="col-sm-8 mt-3 mt-md-0">
    {% include figure.liquid 
      loading="lazy" 
      path="assets/img/charges_by_smoker.jpg" 
      title="boxplot" 
      class="img-fluid rounded z-depth-1"
    %}
  </div>

  <div class="col-sm-8 mt-3 mt-md-4">
    {% include figure.liquid 
      loading="lazy" 
      path="assets/img/bmi_vs_charges.jpg" 
      title="scatterplot" 
      class="img-fluid rounded z-depth-1"
    %}
  </div>
</div>

<div class="caption text-center mt-2">
    Figura 3. Arriba: Boxplot representado charges si el usuario es fumador o no | Abajo: Mapa de puntos represando el IMC vs gastos si el usuario es fumador o no fumador.
</div>

- Relaciones clave: se exploró visualmente la relación entre variables como edad, BMI y costo del seguro para detectar tendencias o comportamientos relevantes. 

Este análisis permitió identificar los factores con mayor impacto en los costos y guiar la construcción de los modelos predictivos.

### 2. Procesamiento y preparación de los datos

Con el propósito de asegurar que los modelos de Machine Learning funcionaran de manera óptima, se aplicaron las siguientes transformaciones y pasos de preparación:

- Codificación de variables categóricas: se utilizó One-Hot Encoding para convertir las categorías de variables como sex, smoker y region en variables numéricas.

```python
df_encoded = pd.get_dummies(df, drop_first=True)
```
- División del dataset: el conjunto de datos se dividió en datos de entrenamiento (80%) y prueba (20%) para evaluar el rendimiento real de los modelos.

```python
# Division Train/test

from sklearn.model_selection import train_test_split

x = df_encoded.drop('charges', axis=1)
y = df_encoded['charges']

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
```

- Estandarización de características: se aplicó StandardScaler a las variables numéricas para normalizar su escala, lo cual mejora el desempeño de modelos como Regresión Lineal y K-Nearest Neighbors.

```python
# Escalada de variables 

from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
x_train_scaled = scaler.fit_transform(x_train)
x_test_scaled = scaler.transform(x_test)
```

- Eliminación y manejo de outliers: se identificaron valores extremos mediante boxplots; debido a que algunos outliers son representativos, especialmente para la columna charges, se conservaron para no perder información relevante. 
- Construcción de matrices de características **x** y objetivo **y**: se organizo el dataset en variables predictoras y la variable charges a predecir. 

Esta etapa dejó el data set preparado para entrenar los modelos de predicción, asegurando coherencia y calidad en los datos utilizados. 

</div>

<hr>

## Modelos 

<div class="text-justify" markdown="1">

En este proyecto se implementaron tres modelos supervisados de regresión: **Regresión lineal**, **Árbol de Decisión** y **K-Nearest Neighbors.**
Cada uno ofrece un enfoque distinto para predecir el costo de los seguros médicos con base a las características demográficas y de salud, permitiendo comparar su desempeño y comprender mejor cómo cada técnica captura las relaciones presentes en los datos.

<hr>

### 1. Regresión lineal 

La **Regresión Lineal** es uno de los modelos más utilizados en problemas de predicción númerica. Busca encontrar una relación lineal entre las variables independientes (edad, BMI, número de hijos y región) y la variable objetivo que es charges.

El modelo estima una ecuación del tipo: 

$$
y = \beta_0 + \beta_1 x_1 + \beta_2 x_2 + \cdots + \beta_n x_n
$$


**¿Por qué utilizarla aquí?**
- Permite identificar qué variables influyen más en el costo del seguro mediante sus coeficientes.
- Sirve como línea de base simple para comparar con modelos más complejos.
- Funciona bien cuando las relaciones entre variables son aproximadamente lineales, como ocurren en ciertos segmentos del dataset (edad y BMI muestran tendencias claras con respecto al costo).

**Ventajas**
- Fácil interpretación.
- Entrenamiento rápido. 
- Buena para explicar relaciones entre variables. 

<hr>

### 2. Árbol de Decisión (Decesion Tree Regressor)

El modelo de **Árbol de Decisión** divide recursivamente los datos en segmentos basados en reglas del tipo: 

> "Si BMI > 30 y el paciente es fumador, entonces el costo tiende a ser alto"

Este modelo no busca una relación lineal, sino que aprende reglas y patrones complejos que describen los datos.

**¿Por qué utilizarlo?**
- Captura relaciones no líneales entre variables.
- Permite modelar interacciones naturales en el dataset. Por ejemplo: fumador + BMI Alto → costos muy elevados.
- Es fácil de visualizar y explicar con reglas. 

**Ventajas** 
- No requiere escalado de variables. 
- Interprete mediante su estructura en forma de árbol.
- Capaz de capturar relaciones complejas que la regresión lineal no puede identificar. 

<hr>

### 3. K-Nearest Neighbors 

El modelo **K-Nearest Neighbors** predice el costo del seguro basándose en los valores de los **k vecinos** más cercanos en el espacio de variables. 

Ejemplo: 

> Para un paciente nuevo, busca otros pacientes similares (vecinos) y promedia sus costos.

**¿Por qupe utilizarlo?**

- Funciona muy bien para patrones complejos que no siguen reglas lineales. 
- Es un modelo basado en similitud: ideal para datasets donde personas con características similares tienden a tener costos parecidos.
- Permite evaluar cómo las distancias entre caracter´siticas influyen en la predicción.

**REQUISITO IMPORTANTE**

El dataset debe estar escalado, es por eso que se usó *StandardScaler*, debido a que K-Nearest Neighbors depende de distancias y variables con diferentes unidades podrían sesgar el cálculo.

**Ventajas**
- No asume ninguna forma funcional del modelo.
- Puede lograr alto desmpeño con datos bien procesados.
- Ideal para comparar con modelos basados en regalas o ecuaciones.

<hr>

### Conclusiones del uso de los modelos Regresión Lineal, Árbol de decisiones y K-Nearest Neighbors

El uso de estos 3 modelos permite: 
- Obetener una visión líneal de la relación entre características y costos (**Regresión Lineal**)
- capturar reglas no lineales y comportamientos excepcionales (**Árbol de Decisión**).
- Predecir costos basandose en patrones de similitud entre pacientes (**K-Nearest Neighbors**)

Esta combinación ofrece un análisis robusto y completo del comportamiento del dataset, permitiendo identificar qué metodos funcionan mejor en la predicción del costo real de los seguros médicos.

</div>

<hr>

## Resultados

<div class="text-justify" markdown="1">

Los resultados del proyecto se basan en la comparación del desempeño de tres modelos de regresión: **Regresión Lineal, Árbol de Decisión y K-Nearest Neighbors.** Para evaluar cada modelo, primero se prepararon los datos adecuadamente y después se calcularon métricas de desempeño como **R² y MAE**, que permiten medir qué tan bien cada algortimo logra predecir el costo del seguro.

A continuación se muestra el desempeño de cada algoritmo después del entrenamiento y evaluación sobre los datos prueba. 

### 1. Regresión Lineal 

La Regresión Lineal busca establecer la relación entre variables predictoras y el costo del seguro mediante una combinación lineal de las caracteristicas. 

**Métricas obtenidas:**
-**R²**: Mide la proporción de variabilidad explicada por el modelo.
-**MAE**: Mide el error absoluto promedio


```python
# Regresión Lineal
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

lr = LinearRegression()
lr.fit(x_train_scaled, y_train)
y_pred_lr = lr.predict(x_test_scaled)

print('Linear Regression:')
print(f'MAE: {mean_absolute_error(y_test, y_pred_lr)}')
print(f'MSE: {mean_squared_error(y_test, y_pred_lr)}')
print(f'R2: {r2_score(y_test, y_pred_lr)}')
```
El modelo de Regresión Lineal obtuvo: 
- **R²**: 0.7836
- **MAE**: 4181.19

Este desempeño indica que el modelo logra explicar alrededor del 78% de la variabilidad del costo del seguro médico a partir de las variables predictoras. Sin embargo, su MAE es el más alto de los tres modelos, lo cual significa que, aunque el modelo captura bien la tendencia general de los datos, tiene dificultades para oredecir casos con valores extremos, por ejemplo, costos muy altos asociados con tabaquísmo o alto BMI.

Esto es coherente con que la Regresión Lineal asume relaciones lineales y no se ajusta de manera optima a relaciones no lineales presentes en el dataset.

<hr>

### 2. Decision Tree (Árbol de Decisión)

Este modelo divide los datos en nodos basados en reglas que intentan separar grupos con costos similares. 

Es útil porque detecta relaciones no lineales sin requerir el escalado de variables.

```python
# Arbol de decisión 
from sklearn.tree import DecisionTreeRegressor

tree = DecisionTreeRegressor(max_depth=5, random_state=42)
tree.fit(x_train, y_train)

y_pred_tree = tree.predict(x_test)

print('Decision Tree:')
print(f'MAE: {mean_absolute_error(y_test, y_pred_tree)}')
print(f'MSE: {mean_squared_error(y_test, y_pred_tree)}')
print(f'R2: {r2_score(y_test, y_pred_lr)}')
```

El modelo mostro: 
- **R²**: 0.7836
- **MAE**: 2930.77

Su R² es exactamente igual al de la Regresión Lineal, pero presenta un MAE mucho menor, lo que indica que este modelo logra ajustarse mejor a patrones no lineales y reduce significativamente el error promedio de sus predicciones. 

Los Árboles de Decisión suelen capturar interacciones entre variables y divisiones complejas dentro de los datos, lo que explica su superioridad en MAE. Sin embargo, este tipo de modelo puede sobreajustar si no se regula, aunque en este caso su R² es moderado y consistente.

<hr>

### 3. K-Nearest Neighbors

 K-Nearest Neighbors predice el costo del seguro básandose en la información de los **k** individuos más cercanos en el espacio de características. 

 Para este modelo, la normalizació fue escencial para evitar sesgos por escala.

```python
# K-Nearest Neighbors 
from sklearn.neighbors import KNeighborsRegressor

knn = KNeighborsRegressor(n_neighbors=5)
knn.fit(x_train_scaled, y_train)

y_pred_knn = knn.predict(x_test_scaled)

print("KNN:")
print("MAE:", mean_absolute_error(y_test, y_pred_knn))
print("MSE:", mean_squared_error(y_test, y_pred_knn))
print("R2:", r2_score(y_test, y_pred_knn))
```

El modelo mostro: 
- **R²**: 0.8038 (**El más alto**)
- **MAE**: 3494.75

Este modelo es el que mejor explica la variabilidad del costo del seguro, alcanzando un R² superior al 80%. El MAE, aunque mayor que el del Árbol de Decisión, sigue siendo considerablemente más bajo que el de la Regresión Lineal. 

K-Nearest Neighbors tiende a funcionar muy bien cuando hay relaciones complejas entre variables y cuando los datos han sido correctamente estandarizados. Su alto R² indica que logra capturar patrones fines en los datos basándose en vecindades locales.

La gráfica siguiente presenta de forma visual los valores de **MAE** y **R²** para cada modelado evaluado.

<div class="row justify-content-center">
  <div class="col-sm-8 mt-3 mt-md-4">
    {% include figure.liquid loading="eager" path="assets/img/error_models.jpg" title="Error de modelos" class="img-fluid rounded z-depth-1" %}
  </div>
</div>
<div class="caption text-center mt-2">
    Figura 4. Izquierda: Histograma de error MAE por cada modelo. | Derecha: Histograma de error R² por cada modelo.
</div>

</div>

<hr>

## Conclusiones

<div class="text-justify" markdown="1">


En conjunto, los resultados muestran que los modelos **no lineales** superan claramente a la Regresión Lineal, lo que sugiere que la relación entre las variables del dataset y el costo del seguro **no es lineal**.
**K-Nearest Neighbors** es el modelo con mejor capacidad explicativa (mayor valor de R²), lo que significa que es el que mejor captura los patrones generales del costo de los seguros, por otro lado, **Decision Tree** es el modelo más preciso segpun el MAE, ya que comete el menor error promedio en las predicciones.

Los resultados permiten concluir que el dataset presentan relaciones complejas y no líneales, lo cual hace que modelos basados en vecinos o decisiones sean más adecuados que modelos lineales tradicionales.

</div>