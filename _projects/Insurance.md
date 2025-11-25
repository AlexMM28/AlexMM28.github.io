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
- Evaluar los modelos utilizando métricas como MAE, MSE y R².
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
