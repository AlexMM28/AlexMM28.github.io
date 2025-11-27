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

## Modelos 

<div class="text-justify" markdown="1">

En este proyecto se implementaron tres modelos supervisados de regresión: **Regresión lineal**, **Árbol de Decisión** y **K-Nearest Neighbors.**
Cada uno ofrece un enfoque distinto para predecir el costo de los seguros médicos con base a las características demográficas y de salud, permitiendo comparar su desempeño y comprender mejor cómo cada técnica captura las relaciones presentes en los datos.

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

<br>

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

<br>

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

<br>

### Conclusiones del uso de los modelos Regresión Lineal, Árbol de decisiones y K-Nearest Neighbors

El uso de estos 3 modelos permite: 
- Obetener una visión líneal de la relación entre características y costos (**Regresión Lineal**)
- capturar reglas no lineales y comportamientos excepcionales (**Árbol de Decisión**).
- Predecir costos basandose en patrones de similitud entre pacientes (**K-Nearest Neighbors**)

Esta combinación ofrece un análisis robusto y completo del comportamiento del dataset, permitiendo identificar qué metodos funcionan mejor en la predicción del costo real de los seguros médicos.

</div>