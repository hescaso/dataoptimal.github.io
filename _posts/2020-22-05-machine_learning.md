---
title: "Machine Learning Dataset Vinícola"
date: 2020-05-22
tags: [machine learning, regression, classification]
header:
image: "/images/MachineLearning/champan.jpg"
excerpt: "Data Science, Machine Learning"
classes: "wide"
mathjax: "true"

---

# Clasificación y Regresión

Vamos a utilizar un dataset sobre distintos vinos con sus características (como pueden ser la acidez, densidad...). Tendremos que generar, entrenar, validar y testear modelos tanto de clasificación como de regresión.

El dataset proviene de la Universdad de Minho, generado por [P. Cortez](http://www3.dsi.uminho.pt/pcortez/Home.html) et al. Dicho dataset se encuentra en el [*UC Irvine Machine Learning Repository*](https://archive.ics.uci.edu/ml/index.html) ([aquí](https://archive.ics.uci.edu/ml/datasets/Wine+Quality) está disponible).

Adjunto la descripción del dataset:

*Citation Request:
  This dataset is public available for research. The details are described in [Cortez et al., 2009]. 
  Please include this citation if you plan to use this database:*

*Available at: [@Elsevier] http://dx.doi.org/10.1016/j.dss.2009.05.016
                [Pre-press (pdf)] http://www3.dsi.uminho.pt/pcortez/winequality09.pdf
                [bib] http://www3.dsi.uminho.pt/pcortez/dss09.bib*

1. Title: Wine Quality 

2. Created by: Paulo Cortez (Univ. Minho), Antonio Cerdeira, Fernando Almeida, Telmo Matos and Jose Reis (CVRVV) @ 2009
   
3. Past Usage:

  P. Cortez, A. Cerdeira, F. Almeida, T. Matos and J. Reis. Modeling wine preferences by data mining from physicochemical properties.
  In Decision Support Systems, Elsevier, 47(4):547-553. ISSN: 0167-9236.
 
4. Relevant Information:

   The two datasets are related to red and white variants of the Portuguese "Vinho Verde" wine.
   For more details, consult: http://www.vinhoverde.pt/en/ or the reference [Cortez et al., 2009].
   Due to privacy and logistic issues, only physicochemical (inputs) and sensory (the output) variables 
   are available (e.g. there is no data about grape types, wine brand, wine selling price, etc.).

5. Number of Instances: red wine - 1599; white wine - 4898. 

6. Number of Attributes: 11 + output attribute
  
7. Attribute information:

   For more information, read [Cortez et al., 2009].

   Input variables (based on physicochemical tests):
   1 - fixed acidity
   2 - volatile acidity
   3 - citric acid
   4 - residual sugar
   5 - chlorides
   6 - free sulfur dioxide
   7 - total sulfur dioxide
   8 - density
   9 - pH
   10 - sulphates
   11 - alcohol
   Output variable (based on sensory data): 
   12 - quality (score between 0 and 10)

8. Missing Attribute Values: None


Además de las 12 variables descritas, el dataset tiene otra: si el vino es blanco o rojo. Dicho esto, los objetivos son:

1. Separar el dataset en training (+ validación si no vas a hacer validación cruzada) y testing, haciendo antes (o después) las transformaciones de los datos que consideres oportunas, así como selección de variables, reducción de dimensionalidad... Puede que decidas usar los datos tal cual vienen también...
2. Hacer un modelo capaz de clasificar lo mejor posible si un vino es blanco o rojo a partir del resto de variables (vas a ver que está chupado conseguir un muy buen resultado).
3. Hacer un modelo regresor que prediga lo mejor posible la calidad de los vinos.

¡Vamos a ello!


```python
# Comenzamos importando numpy, pandas y matplotlib:
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
```


```python
# Leemos el archivo:
df = pd.read_csv("C:\\winequality.csv", sep = ";", encoding ='utf-8')
df.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>fixed acidity</th>
      <th>volatile acidity</th>
      <th>citric acid</th>
      <th>residual sugar</th>
      <th>chlorides</th>
      <th>free sulfur dioxide</th>
      <th>total sulfur dioxide</th>
      <th>density</th>
      <th>pH</th>
      <th>sulphates</th>
      <th>alcohol</th>
      <th>quality</th>
      <th>color</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>0</td>
      <td>5.20</td>
      <td>0.34</td>
      <td>0.00</td>
      <td>1.8</td>
      <td>0.050</td>
      <td>27.0</td>
      <td>63.0</td>
      <td>0.99160</td>
      <td>3.68</td>
      <td>0.79</td>
      <td>14.0</td>
      <td>6</td>
      <td>red</td>
    </tr>
    <tr>
      <td>1</td>
      <td>6.20</td>
      <td>0.55</td>
      <td>0.45</td>
      <td>12.0</td>
      <td>0.049</td>
      <td>27.0</td>
      <td>186.0</td>
      <td>0.99740</td>
      <td>3.17</td>
      <td>0.50</td>
      <td>9.3</td>
      <td>6</td>
      <td>white</td>
    </tr>
    <tr>
      <td>2</td>
      <td>7.15</td>
      <td>0.17</td>
      <td>0.24</td>
      <td>9.6</td>
      <td>0.119</td>
      <td>56.0</td>
      <td>178.0</td>
      <td>0.99578</td>
      <td>3.15</td>
      <td>0.44</td>
      <td>10.2</td>
      <td>6</td>
      <td>white</td>
    </tr>
    <tr>
      <td>3</td>
      <td>6.70</td>
      <td>0.64</td>
      <td>0.23</td>
      <td>2.1</td>
      <td>0.080</td>
      <td>11.0</td>
      <td>119.0</td>
      <td>0.99538</td>
      <td>3.36</td>
      <td>0.70</td>
      <td>10.9</td>
      <td>5</td>
      <td>red</td>
    </tr>
    <tr>
      <td>4</td>
      <td>7.60</td>
      <td>0.23</td>
      <td>0.34</td>
      <td>1.6</td>
      <td>0.043</td>
      <td>24.0</td>
      <td>129.0</td>
      <td>0.99305</td>
      <td>3.12</td>
      <td>0.70</td>
      <td>10.4</td>
      <td>5</td>
      <td>white</td>
    </tr>
  </tbody>
</table>
</div>



# A.- MODELO DE CLASIFICACIÓN


```python
# Vemos cuantas observaciones hay de cada tipo en la variable "color":
print(df["color"].value_counts())
```

    white    4898
    red      1599
    Name: color, dtype: int64
    

Vemos que aproximadamente el 75% de las observaciones son de vino blanco, y casi el 25% de vino tinto.
Con estas proporciones no podemos utilizar la métrica accuracy, por lo que tendremos que utilizar otra métrica. 


```python
# Creamos listas con todas las columnas del dataset, y de las features del problema de clasificación:
nombres_columnas = ["fixed acidity","volatile acidity","citric acid","residual sugar","chlorides","free sulfur dioxide",
                    "total sulfur dioxide","density","pH","sulphates","alcohol","quality","color"]
nombres_features = ["fixed acidity","volatile acidity","citric acid","residual sugar","chlorides","free sulfur dioxide",
                    "total sulfur dioxide","density","pH","sulphates","alcohol","quality"]
```


```python
# Como la variable a predecir "color" viene como tipo str, vamos a transformarla en una variable numérica:
from sklearn.preprocessing import LabelEncoder
label_encoder = LabelEncoder()
label_encoder.fit(df["color"])
df["color"] = label_encoder.transform(df["color"])

# Y comprobamos que ha funcionado:
df.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>fixed acidity</th>
      <th>volatile acidity</th>
      <th>citric acid</th>
      <th>residual sugar</th>
      <th>chlorides</th>
      <th>free sulfur dioxide</th>
      <th>total sulfur dioxide</th>
      <th>density</th>
      <th>pH</th>
      <th>sulphates</th>
      <th>alcohol</th>
      <th>quality</th>
      <th>color</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>0</td>
      <td>5.20</td>
      <td>0.34</td>
      <td>0.00</td>
      <td>1.8</td>
      <td>0.050</td>
      <td>27.0</td>
      <td>63.0</td>
      <td>0.99160</td>
      <td>3.68</td>
      <td>0.79</td>
      <td>14.0</td>
      <td>6</td>
      <td>0</td>
    </tr>
    <tr>
      <td>1</td>
      <td>6.20</td>
      <td>0.55</td>
      <td>0.45</td>
      <td>12.0</td>
      <td>0.049</td>
      <td>27.0</td>
      <td>186.0</td>
      <td>0.99740</td>
      <td>3.17</td>
      <td>0.50</td>
      <td>9.3</td>
      <td>6</td>
      <td>1</td>
    </tr>
    <tr>
      <td>2</td>
      <td>7.15</td>
      <td>0.17</td>
      <td>0.24</td>
      <td>9.6</td>
      <td>0.119</td>
      <td>56.0</td>
      <td>178.0</td>
      <td>0.99578</td>
      <td>3.15</td>
      <td>0.44</td>
      <td>10.2</td>
      <td>6</td>
      <td>1</td>
    </tr>
    <tr>
      <td>3</td>
      <td>6.70</td>
      <td>0.64</td>
      <td>0.23</td>
      <td>2.1</td>
      <td>0.080</td>
      <td>11.0</td>
      <td>119.0</td>
      <td>0.99538</td>
      <td>3.36</td>
      <td>0.70</td>
      <td>10.9</td>
      <td>5</td>
      <td>0</td>
    </tr>
    <tr>
      <td>4</td>
      <td>7.60</td>
      <td>0.23</td>
      <td>0.34</td>
      <td>1.6</td>
      <td>0.043</td>
      <td>24.0</td>
      <td>129.0</td>
      <td>0.99305</td>
      <td>3.12</td>
      <td>0.70</td>
      <td>10.4</td>
      <td>5</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Vemos el largo del dataframe
len(df)
```




    6497




```python
# Importamos la librería que vamos a utilizar para separar el train del test.
from sklearn.model_selection import train_test_split
```


```python
# Separamos aleatoriamente el data set en un 80% para train y un 20% 
# para test (con ultimo argumento semilla para poder replicar el experimento).
df_separado = train_test_split(df, train_size = 0.8, test_size = 0.2, random_state = 8)
```


```python
# Creamos dos datasets separados que contengan el train y el test:
df_train = df_separado[0]
df_test = df_separado[1]
```


```python
# Comprobamos que se han creado correctamente:
print(df_train.head())
print(df_test.head())
```

          fixed acidity  volatile acidity  citric acid  residual sugar  chlorides  \
    2406            6.0              0.22         0.23             5.0      0.045   
    4847           10.2              0.36         0.64             2.9      0.122   
    6107            7.3              0.26         0.31             1.6      0.040   
    2082            6.6              0.50         0.26            11.3      0.029   
    2601            6.3              0.40         0.24             5.1      0.036   
    
          free sulfur dioxide  total sulfur dioxide  density    pH  sulphates  \
    2406                 10.0                 122.0  0.99261  2.94       0.63   
    4847                 10.0                  41.0  0.99800  3.23       0.66   
    6107                 39.0                 173.0  0.99180  3.19       0.51   
    2082                 32.0                 110.0  0.99302  3.27       0.78   
    2601                 43.0                 131.0  0.99186  3.24       0.44   
    
          alcohol  quality  color  
    2406     10.0        6      1  
    4847     12.5        6      0  
    6107     11.4        6      1  
    2082     12.9        8      1  
    2601     11.3        6      1  
          fixed acidity  volatile acidity  citric acid  residual sugar  chlorides  \
    2598            6.8             0.210         0.36           18.10      0.046   
    590             7.5             0.705         0.24            1.80      0.360   
    6234            7.9             0.440         0.26            4.45      0.033   
    2737            7.3             0.400         0.30            1.70      0.080   
    2802            6.8             0.360         0.24            4.60      0.039   
    
          free sulfur dioxide  total sulfur dioxide  density    pH  sulphates  \
    2598                 32.0                 133.0  1.00000  3.27       0.48   
    590                  15.0                  63.0  0.99640  3.00       1.59   
    6234                 23.0                 100.0  0.99117  3.17       0.52   
    2737                 33.0                  79.0  0.99690  3.41       0.65   
    2802                 24.0                 124.0  0.99090  3.27       0.34   
    
          alcohol  quality  color  
    2598      8.8        5      1  
    590       9.5        5      0  
    6234     12.7        6      1  
    2737      9.5        6      0  
    2802     12.6        7      1  
    


```python
print("Conjunto train tiene: ",len(df_train))
print("Conjunto test tiene: ",len(df_test))
```

    Conjunto train tiene:  5197
    Conjunto test tiene:  1300
    

# ANÁLISIS EXPLORATORIO


```python
df_train.describe()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>fixed acidity</th>
      <th>volatile acidity</th>
      <th>citric acid</th>
      <th>residual sugar</th>
      <th>chlorides</th>
      <th>free sulfur dioxide</th>
      <th>total sulfur dioxide</th>
      <th>density</th>
      <th>pH</th>
      <th>sulphates</th>
      <th>alcohol</th>
      <th>quality</th>
      <th>color</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>count</td>
      <td>5197.000000</td>
      <td>5197.000000</td>
      <td>5197.000000</td>
      <td>5197.000000</td>
      <td>5197.000000</td>
      <td>5197.000000</td>
      <td>5197.000000</td>
      <td>5197.000000</td>
      <td>5197.000000</td>
      <td>5197.000000</td>
      <td>5197.000000</td>
      <td>5197.000000</td>
      <td>5197.000000</td>
    </tr>
    <tr>
      <td>mean</td>
      <td>7.216933</td>
      <td>0.340332</td>
      <td>0.317458</td>
      <td>5.407283</td>
      <td>0.056127</td>
      <td>30.312680</td>
      <td>115.079758</td>
      <td>0.994693</td>
      <td>3.217289</td>
      <td>0.531932</td>
      <td>10.484258</td>
      <td>5.819896</td>
      <td>0.751203</td>
    </tr>
    <tr>
      <td>std</td>
      <td>1.298290</td>
      <td>0.165424</td>
      <td>0.145665</td>
      <td>4.670692</td>
      <td>0.035189</td>
      <td>17.655266</td>
      <td>56.489105</td>
      <td>0.002938</td>
      <td>0.162411</td>
      <td>0.148217</td>
      <td>1.186195</td>
      <td>0.865850</td>
      <td>0.432358</td>
    </tr>
    <tr>
      <td>min</td>
      <td>4.200000</td>
      <td>0.080000</td>
      <td>0.000000</td>
      <td>0.600000</td>
      <td>0.012000</td>
      <td>1.000000</td>
      <td>6.000000</td>
      <td>0.987220</td>
      <td>2.740000</td>
      <td>0.230000</td>
      <td>8.000000</td>
      <td>3.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <td>25%</td>
      <td>6.400000</td>
      <td>0.230000</td>
      <td>0.240000</td>
      <td>1.800000</td>
      <td>0.038000</td>
      <td>17.000000</td>
      <td>76.000000</td>
      <td>0.992400</td>
      <td>3.110000</td>
      <td>0.430000</td>
      <td>9.500000</td>
      <td>5.000000</td>
      <td>1.000000</td>
    </tr>
    <tr>
      <td>50%</td>
      <td>7.000000</td>
      <td>0.290000</td>
      <td>0.310000</td>
      <td>3.000000</td>
      <td>0.047000</td>
      <td>28.000000</td>
      <td>118.000000</td>
      <td>0.994880</td>
      <td>3.200000</td>
      <td>0.510000</td>
      <td>10.300000</td>
      <td>6.000000</td>
      <td>1.000000</td>
    </tr>
    <tr>
      <td>75%</td>
      <td>7.700000</td>
      <td>0.410000</td>
      <td>0.390000</td>
      <td>8.100000</td>
      <td>0.065000</td>
      <td>41.000000</td>
      <td>155.000000</td>
      <td>0.997000</td>
      <td>3.320000</td>
      <td>0.600000</td>
      <td>11.300000</td>
      <td>6.000000</td>
      <td>1.000000</td>
    </tr>
    <tr>
      <td>max</td>
      <td>15.900000</td>
      <td>1.580000</td>
      <td>1.660000</td>
      <td>31.600000</td>
      <td>0.611000</td>
      <td>146.500000</td>
      <td>344.000000</td>
      <td>1.010300</td>
      <td>4.010000</td>
      <td>2.000000</td>
      <td>14.900000</td>
      <td>9.000000</td>
      <td>1.000000</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Vamos a graficar con Matplotlib. 
%matplotlib inline

# También importamos la biblioteca "Seaborn":
import seaborn as sns
```


```python
# Gráfico boxplot:
df_train.plot(kind = "box", figsize = (22, 10))
```




    <matplotlib.axes._subplots.AxesSubplot at 0x1cfcb756bc8>


{% include figure image_path="/images/MachineLearning/output_19_1.png" %}


Vemos como las variables tienen distintos valores, por lo que va a ser necesario hacer una estandarización de las variables.

También vemos que las variables tienen outliers positivos. Estos no los vamos a quitar ya que pueden aportar información a nuestro modelo.


```python
# Dibujamos una matriz scatter plot, con histograma (KDE)
sns.pairplot(df_train, hue = "color", height = 2)
```

    C:\Users\hesca\Anaconda3\lib\site-packages\statsmodels\nonparametric\kde.py:487: RuntimeWarning: invalid value encountered in true_divide
      binned = fast_linbin(X, a, b, gridsize) / (delta * nobs)
    C:\Users\hesca\Anaconda3\lib\site-packages\statsmodels\nonparametric\kdetools.py:34: RuntimeWarning: invalid value encountered in double_scalars
      FAC1 = 2*(np.pi*bw/RANGE)**2
    C:\Users\hesca\Anaconda3\lib\site-packages\statsmodels\nonparametric\kde.py:487: RuntimeWarning: invalid value encountered in true_divide
      binned = fast_linbin(X, a, b, gridsize) / (delta * nobs)
    C:\Users\hesca\Anaconda3\lib\site-packages\statsmodels\nonparametric\kdetools.py:34: RuntimeWarning: invalid value encountered in double_scalars
      FAC1 = 2*(np.pi*bw/RANGE)**2
    




    <seaborn.axisgrid.PairGrid at 0x1cfc3619208>



{% include figure image_path="/images/MachineLearning/output_21_2.png" %}

```python
# Sacamos la matriz de correlación:
df_train.corr()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>fixed acidity</th>
      <th>volatile acidity</th>
      <th>citric acid</th>
      <th>residual sugar</th>
      <th>chlorides</th>
      <th>free sulfur dioxide</th>
      <th>total sulfur dioxide</th>
      <th>density</th>
      <th>pH</th>
      <th>sulphates</th>
      <th>alcohol</th>
      <th>quality</th>
      <th>color</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>fixed acidity</td>
      <td>1.000000</td>
      <td>0.213234</td>
      <td>0.328516</td>
      <td>-0.120155</td>
      <td>0.300877</td>
      <td>-0.289083</td>
      <td>-0.334544</td>
      <td>0.461900</td>
      <td>-0.247001</td>
      <td>0.304669</td>
      <td>-0.083342</td>
      <td>-0.074363</td>
      <td>-0.486519</td>
    </tr>
    <tr>
      <td>volatile acidity</td>
      <td>0.213234</td>
      <td>1.000000</td>
      <td>-0.382472</td>
      <td>-0.205666</td>
      <td>0.373642</td>
      <td>-0.353704</td>
      <td>-0.416502</td>
      <td>0.271000</td>
      <td>0.272074</td>
      <td>0.223726</td>
      <td>-0.042430</td>
      <td>-0.276720</td>
      <td>-0.652588</td>
    </tr>
    <tr>
      <td>citric acid</td>
      <td>0.328516</td>
      <td>-0.382472</td>
      <td>1.000000</td>
      <td>0.140537</td>
      <td>0.049010</td>
      <td>0.136431</td>
      <td>0.197047</td>
      <td>0.091437</td>
      <td>-0.337806</td>
      <td>0.065515</td>
      <td>-0.007286</td>
      <td>0.083195</td>
      <td>0.190788</td>
    </tr>
    <tr>
      <td>residual sugar</td>
      <td>-0.120155</td>
      <td>-0.205666</td>
      <td>0.140537</td>
      <td>1.000000</td>
      <td>-0.138750</td>
      <td>0.422160</td>
      <td>0.504632</td>
      <td>0.535733</td>
      <td>-0.278265</td>
      <td>-0.189503</td>
      <td>-0.368322</td>
      <td>-0.037540</td>
      <td>0.355706</td>
    </tr>
    <tr>
      <td>chlorides</td>
      <td>0.300877</td>
      <td>0.373642</td>
      <td>0.049010</td>
      <td>-0.138750</td>
      <td>1.000000</td>
      <td>-0.206302</td>
      <td>-0.285436</td>
      <td>0.362698</td>
      <td>0.040886</td>
      <td>0.407305</td>
      <td>-0.252015</td>
      <td>-0.199100</td>
      <td>-0.511351</td>
    </tr>
    <tr>
      <td>free sulfur dioxide</td>
      <td>-0.289083</td>
      <td>-0.353704</td>
      <td>0.136431</td>
      <td>0.422160</td>
      <td>-0.206302</td>
      <td>1.000000</td>
      <td>0.723644</td>
      <td>0.027797</td>
      <td>-0.157090</td>
      <td>-0.193942</td>
      <td>-0.179858</td>
      <td>0.061717</td>
      <td>0.481437</td>
    </tr>
    <tr>
      <td>total sulfur dioxide</td>
      <td>-0.334544</td>
      <td>-0.416502</td>
      <td>0.197047</td>
      <td>0.504632</td>
      <td>-0.285436</td>
      <td>0.723644</td>
      <td>1.000000</td>
      <td>0.024362</td>
      <td>-0.251069</td>
      <td>-0.282675</td>
      <td>-0.265596</td>
      <td>-0.035733</td>
      <td>0.709680</td>
    </tr>
    <tr>
      <td>density</td>
      <td>0.461900</td>
      <td>0.271000</td>
      <td>0.091437</td>
      <td>0.535733</td>
      <td>0.362698</td>
      <td>0.027797</td>
      <td>0.024362</td>
      <td>1.000000</td>
      <td>0.014424</td>
      <td>0.266917</td>
      <td>-0.696727</td>
      <td>-0.309818</td>
      <td>-0.400037</td>
    </tr>
    <tr>
      <td>pH</td>
      <td>-0.247001</td>
      <td>0.272074</td>
      <td>-0.337806</td>
      <td>-0.278265</td>
      <td>0.040886</td>
      <td>-0.157090</td>
      <td>-0.251069</td>
      <td>0.014424</td>
      <td>1.000000</td>
      <td>0.198838</td>
      <td>0.123105</td>
      <td>0.015208</td>
      <td>-0.341926</td>
    </tr>
    <tr>
      <td>sulphates</td>
      <td>0.304669</td>
      <td>0.223726</td>
      <td>0.065515</td>
      <td>-0.189503</td>
      <td>0.407305</td>
      <td>-0.193942</td>
      <td>-0.282675</td>
      <td>0.266917</td>
      <td>0.198838</td>
      <td>1.000000</td>
      <td>-0.004274</td>
      <td>0.033934</td>
      <td>-0.488812</td>
    </tr>
    <tr>
      <td>alcohol</td>
      <td>-0.083342</td>
      <td>-0.042430</td>
      <td>-0.007286</td>
      <td>-0.368322</td>
      <td>-0.252015</td>
      <td>-0.179858</td>
      <td>-0.265596</td>
      <td>-0.696727</td>
      <td>0.123105</td>
      <td>-0.004274</td>
      <td>1.000000</td>
      <td>0.441763</td>
      <td>0.025272</td>
    </tr>
    <tr>
      <td>quality</td>
      <td>-0.074363</td>
      <td>-0.276720</td>
      <td>0.083195</td>
      <td>-0.037540</td>
      <td>-0.199100</td>
      <td>0.061717</td>
      <td>-0.035733</td>
      <td>-0.309818</td>
      <td>0.015208</td>
      <td>0.033934</td>
      <td>0.441763</td>
      <td>1.000000</td>
      <td>0.121391</td>
    </tr>
    <tr>
      <td>color</td>
      <td>-0.486519</td>
      <td>-0.652588</td>
      <td>0.190788</td>
      <td>0.355706</td>
      <td>-0.511351</td>
      <td>0.481437</td>
      <td>0.709680</td>
      <td>-0.400037</td>
      <td>-0.341926</td>
      <td>-0.488812</td>
      <td>0.025272</td>
      <td>0.121391</td>
      <td>1.000000</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Con tanta variable no podemos observar bien la correlación, vamos a realizar un "mapa de calor":
plt.figure(figsize=(22,6))

sns.heatmap(df_train.corr(),
            vmin = -1,
            vmax = 1,
            annot = True,
            cmap = "RdBu_r")
```




    <matplotlib.axes._subplots.AxesSubplot at 0x1cfc49e63c8>



{% include figure image_path="/images/MachineLearning/output_23_1.png" %}



Vemos que la mayor correlación (0,72) se produce entre las variables "total sulfur dioxide" y "free sulfur dioxide", lo cual parece lógico ya que ambas variables están relacionadas con el dioxido de sulfuro del vino. 

Además, la variable a clasificar "color", está fuertemente relacionado (0,71) con la variable "total sulfur dioxide". 

## Pre-procesado del data set

Vamos a plantearnos si debemos preprocesar los datos:

- **Seleccionar variables**: No vemos variables que estén fuertemente correladas, además no conocemos el dataset ni la composición del vino lo suficiente. Parece que las distintas features son parte de la composición química del vino, por lo que no vamos a eliminar ninguna feature.

- **Quitar outliers**: Pensamos que en los otuliers puede residir bastante información, y además los modelos que vamos a utilizar pueden asumir estos, por lo que no los vamos a quitar.

- **Estandarizar features**: Sí vamos a estandarizar las features para todos nuestros modelos, menos en los árboles, sus ensembles y en Naive Bayes.

- **Reducción de la dimensionalidad**: En este caso al tener solo 12 features, no vamos a reducir la dimensionalidad, ya que todos los modelo a aplicar pueden funcionar bien con este número.

**Vamos a probar distintos modelos.**

## 1.- Regresión Logística
Para comenzar probamos una Regresión Logística, con los siguientes pasos: 
- Lo primero que hacemos es estandarizar el train de las observaciones.
- El segundo paso es hacer la regresión logística.
- Para la validación vamos a utilizar "cross validation"

Para todo esto, en este y en el resto de modelos, vamos a utilizar un Pipeline.


```python
# Importamos las librerías necesarias para este proceso:
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler # para standarizar
from sklearn.linear_model import LogisticRegression # regresión logística 
from sklearn.model_selection import GridSearchCV 
```


```python
# Ahora creamos el pipeline:
pipeline_reglog = Pipeline ([("estandarizar", StandardScaler()),
                            ("reglog", LogisticRegression())
                            ])
# Y definismos los hiperparámetros. En este caso vacío al ser una regresión logística:
grid_hyper_reglog ={}

# definimos el flujo de trabajo
gs_reglog = GridSearchCV(pipeline_reglog, 
                        param_grid = grid_hyper_reglog, # en este caso vacío, pero lo incluimos
                        cv=10, # validación mediante cross validation
                        scoring = 'roc_auc', # métrica cuando no está balanceada la variable a clasificar
                        n_jobs=-1,
                        verbose=3) # la cantidad de datos que queremos que aparezcan

```


```python
# lanzamos el entrenamiento del modelo:
gs_reglog.fit(df_train[nombres_features], df_train["color"]) 
```

    Fitting 10 folds for each of 1 candidates, totalling 10 fits
    

    [Parallel(n_jobs=-1)]: Using backend LokyBackend with 8 concurrent workers.
    [Parallel(n_jobs=-1)]: Done   3 out of  10 | elapsed:    3.6s remaining:    8.6s
    [Parallel(n_jobs=-1)]: Done   7 out of  10 | elapsed:    3.6s remaining:    1.5s
    [Parallel(n_jobs=-1)]: Done  10 out of  10 | elapsed:    3.8s finished
    C:\Users\hesca\Anaconda3\lib\site-packages\sklearn\linear_model\logistic.py:432: FutureWarning: Default solver will be changed to 'lbfgs' in 0.22. Specify a solver to silence this warning.
      FutureWarning)
    




    GridSearchCV(cv=10, error_score='raise-deprecating',
                 estimator=Pipeline(memory=None,
                                    steps=[('estandarizar',
                                            StandardScaler(copy=True,
                                                           with_mean=True,
                                                           with_std=True)),
                                           ('reglog',
                                            LogisticRegression(C=1.0,
                                                               class_weight=None,
                                                               dual=False,
                                                               fit_intercept=True,
                                                               intercept_scaling=1,
                                                               l1_ratio=None,
                                                               max_iter=100,
                                                               multi_class='warn',
                                                               n_jobs=None,
                                                               penalty='l2',
                                                               random_state=None,
                                                               solver='warn',
                                                               tol=0.0001,
                                                               verbose=0,
                                                               warm_start=False))],
                                    verbose=False),
                 iid='warn', n_jobs=-1, param_grid={}, pre_dispatch='2*n_jobs',
                 refit=True, return_train_score=False, scoring='roc_auc',
                 verbose=3)




```python
# Para ver el acierto de nuestro modelo con crossvalidation:
gs_reglog.best_score_
```




    0.9975465346907603



## 2.- Árboles de decisión 


```python
# Importamos la librería necesaria:
from sklearn.tree import DecisionTreeClassifier
```


```python
# Definimos pipeline:
pipeline_arbolito = Pipeline([("arbolito", DecisionTreeClassifier())]) # Solo un paso ya que no vamos a estandarizar.

# Definimos hiperparámetros:
gryd_hyper_arbolito = {"arbolito__max_depth": [2, 3, 4, 5, 6, 7, 8, 9]}  # Definimos las profundidades para entrenar.

gs_arbolito = GridSearchCV(pipeline_arbolito,
                          param_grid = gryd_hyper_arbolito,
                          cv=10,
                          scoring='roc_auc',
                          n_jobs=-1,
                          verbose=3)
```


```python
# lanzamos el entrenamiento del modelo:
gs_arbolito.fit(df_train[nombres_features], df_train["color"])
```

    Fitting 10 folds for each of 8 candidates, totalling 80 fits
    

    [Parallel(n_jobs=-1)]: Using backend LokyBackend with 8 concurrent workers.
    [Parallel(n_jobs=-1)]: Done  16 tasks      | elapsed:    0.1s
    [Parallel(n_jobs=-1)]: Done  38 out of  80 | elapsed:    0.3s remaining:    0.3s
    [Parallel(n_jobs=-1)]: Done  80 out of  80 | elapsed:    0.7s finished
    




    GridSearchCV(cv=10, error_score='raise-deprecating',
                 estimator=Pipeline(memory=None,
                                    steps=[('arbolito',
                                            DecisionTreeClassifier(class_weight=None,
                                                                   criterion='gini',
                                                                   max_depth=None,
                                                                   max_features=None,
                                                                   max_leaf_nodes=None,
                                                                   min_impurity_decrease=0.0,
                                                                   min_impurity_split=None,
                                                                   min_samples_leaf=1,
                                                                   min_samples_split=2,
                                                                   min_weight_fraction_leaf=0.0,
                                                                   presort=False,
                                                                   random_state=None,
                                                                   splitter='best'))],
                                    verbose=False),
                 iid='warn', n_jobs=-1,
                 param_grid={'arbolito__max_depth': [2, 3, 4, 5, 6, 7, 8, 9]},
                 pre_dispatch='2*n_jobs', refit=True, return_train_score=False,
                 scoring='roc_auc', verbose=3)




```python
# Vemos el acierto de nuestro modelo con crossvalidation:
gs_arbolito.best_score_
```




    0.9829919206849101




```python
# Vemos cual es la mejor profundidad:
gs_arbolito.best_params_
```




    {'arbolito__max_depth': 3}



Vemos que la mejor profundidad es 3

## 3.- K-nearest neighbors

Los pasos son:
- Estandarizar
- Probamos con los siguientes números de vecinos: {1,3,5,7,9,11,13,15}


```python
# Importamos librerías
from sklearn.neighbors import KNeighborsClassifier
```


```python
# Definimos pipeline:
pipeline_vecinos = Pipeline([("est", StandardScaler()),
                            ("vecinos", KNeighborsClassifier())
                            ]) 

# Definimos hiperparámetros:
gryd_hyper_vecinos = {"vecinos__n_neighbors": [1,3,5,7,9,11,13,15],
                     }  

gs_vecinos = GridSearchCV(pipeline_vecinos,
                          param_grid = gryd_hyper_vecinos,
                          cv=10,
                          scoring='roc_auc',
                          n_jobs=-1,
                          verbose=3)
```


```python
gs_vecinos.fit(df_train[nombres_features], df_train["color"])
```

    Fitting 10 folds for each of 8 candidates, totalling 80 fits
    

    [Parallel(n_jobs=-1)]: Using backend LokyBackend with 8 concurrent workers.
    [Parallel(n_jobs=-1)]: Done  16 tasks      | elapsed:    0.2s
    [Parallel(n_jobs=-1)]: Done  80 out of  80 | elapsed:    1.4s finished
    




    GridSearchCV(cv=10, error_score='raise-deprecating',
                 estimator=Pipeline(memory=None,
                                    steps=[('est',
                                            StandardScaler(copy=True,
                                                           with_mean=True,
                                                           with_std=True)),
                                           ('vecinos',
                                            KNeighborsClassifier(algorithm='auto',
                                                                 leaf_size=30,
                                                                 metric='minkowski',
                                                                 metric_params=None,
                                                                 n_jobs=None,
                                                                 n_neighbors=5, p=2,
                                                                 weights='uniform'))],
                                    verbose=False),
                 iid='warn', n_jobs=-1,
                 param_grid={'vecinos__n_neighbors': [1, 3, 5, 7, 9, 11, 13, 15]},
                 pre_dispatch='2*n_jobs', refit=True, return_train_score=False,
                 scoring='roc_auc', verbose=3)




```python
gs_vecinos.best_params_
```




    {'vecinos__n_neighbors': 11}



El mejor parámetro es el de 11 vecinos.


```python
gs_vecinos.best_score_
```




    0.9974152475229384



## 4.- Random Forest

Probamos el ensemble Random Forest.

Tenemos 2 hiperparámetros, el número de árboles y la profundidad de los mismos.
Vamos a probar con:
- n_estimators (nº de árboles) = [50, 100, 250, 500]
- max_depth = vamos a probar todas las profundidades de 1 a 50.


```python
# Importamos librerías:
from sklearn.ensemble import RandomForestClassifier

pipeline_rf = Pipeline([("bosque", RandomForestClassifier())]) # Solo un paso porque no estandarizamos las features.

grid_hyper_rf = {"bosque__n_estimators": [50, 100, 250, 500, 1000],
                "bosque__max_depth": np.arange(1,51)}

gs_rf = GridSearchCV(pipeline_rf,
                    param_grid = grid_hyper_rf,
                    cv=10,
                    scoring="roc_auc",
                    n_jobs=-1) 
```


```python
gs_rf.fit(df_train[nombres_features], df_train["color"])
```




    GridSearchCV(cv=10, error_score='raise-deprecating',
                 estimator=Pipeline(memory=None,
                                    steps=[('bosque',
                                            RandomForestClassifier(bootstrap=True,
                                                                   class_weight=None,
                                                                   criterion='gini',
                                                                   max_depth=None,
                                                                   max_features='auto',
                                                                   max_leaf_nodes=None,
                                                                   min_impurity_decrease=0.0,
                                                                   min_impurity_split=None,
                                                                   min_samples_leaf=1,
                                                                   min_samples_split=2,
                                                                   min_weight_fraction_leaf=0.0,
                                                                   n_est...
                                    verbose=False),
                 iid='warn', n_jobs=-1,
                 param_grid={'bosque__max_depth': array([ 1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15, 16, 17,
           18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34,
           35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50]),
                             'bosque__n_estimators': [50, 100, 250, 500, 1000]},
                 pre_dispatch='2*n_jobs', refit=True, return_train_score=False,
                 scoring='roc_auc', verbose=0)




```python
gs_rf.best_params_
```




    {'bosque__max_depth': 23, 'bosque__n_estimators': 250}



Los mejores parámetros son 250 árboles con una profundidad de 23


```python
gs_rf.best_score_
```




    0.998896036709122



**Nuesto modelo predice con una fiabilidad de casi el 99,89%**

## 5.- Naïve Bayes


```python
# Naïve Bayes con y sin KBest:

from sklearn.naive_bayes import GaussianNB
from sklearn.feature_selection import RFECV, SelectKBest, f_classif

nb = GaussianNB()

nb_kbest = Pipeline(steps=[("kbest",SelectKBest()),
                           ("nb",GaussianNB())])

grid_nb_kbest = {"kbest__score_func": [f_classif],
                 "kbest__k": [1,2,3]
                }

gs_nb = GridSearchCV(nb,
                     {},  # No hay grid
                     cv=10,
                     scoring="accuracy",
                     verbose=1,
                     n_jobs=-1)
              
gs_nb_kbest = GridSearchCV(nb_kbest,
                           grid_nb_kbest,
                           cv=10,
                           scoring="accuracy",
                           verbose=1,
                           n_jobs=-1)
```


```python
gs_nb.fit(df_train[nombres_features], df_train["color"])
```

    Fitting 10 folds for each of 1 candidates, totalling 10 fits
    

    [Parallel(n_jobs=-1)]: Using backend LokyBackend with 8 concurrent workers.
    [Parallel(n_jobs=-1)]: Done   6 out of  10 | elapsed:    3.6s remaining:    2.4s
    [Parallel(n_jobs=-1)]: Done  10 out of  10 | elapsed:    3.7s finished
    




    GridSearchCV(cv=10, error_score='raise-deprecating',
                 estimator=GaussianNB(priors=None, var_smoothing=1e-09), iid='warn',
                 n_jobs=-1, param_grid={}, pre_dispatch='2*n_jobs', refit=True,
                 return_train_score=False, scoring='accuracy', verbose=1)




```python
gs_nb.best_score_
```




    0.9719068693477005




```python
gs_nb_kbest.fit(df_train[nombres_features], df_train["color"])
```

    Fitting 10 folds for each of 3 candidates, totalling 30 fits
    

    [Parallel(n_jobs=-1)]: Using backend LokyBackend with 8 concurrent workers.
    [Parallel(n_jobs=-1)]: Done  15 out of  30 | elapsed:    0.1s remaining:    0.1s
    [Parallel(n_jobs=-1)]: Done  30 out of  30 | elapsed:    0.2s finished
    




    GridSearchCV(cv=10, error_score='raise-deprecating',
                 estimator=Pipeline(memory=None,
                                    steps=[('kbest',
                                            SelectKBest(k=10,
                                                        score_func=<function f_classif at 0x0000020C634CBAF8>)),
                                           ('nb',
                                            GaussianNB(priors=None,
                                                       var_smoothing=1e-09))],
                                    verbose=False),
                 iid='warn', n_jobs=-1,
                 param_grid={'kbest__k': [1, 2, 3],
                             'kbest__score_func': [<function f_classif at 0x0000020C634CBAF8>]},
                 pre_dispatch='2*n_jobs', refit=True, return_train_score=False,
                 scoring='accuracy', verbose=1)




```python
gs_nb_kbest.best_params_
```




    {'kbest__k': 2,
     'kbest__score_func': <function sklearn.feature_selection.univariate_selection.f_classif(X, y)>}




```python
gs_nb_kbest.best_score_
```




    0.9557436982874735



## 6.- GRADIENT BOOSTING TREES


```python
from sklearn.ensemble import GradientBoostingClassifier

gradient_boosting = GradientBoostingClassifier()

grid_gradient_boosting = {"loss": ["deviance"], 
                          "learning_rate": [0.05, 0.1, 0.5], 
                          
                          "n_estimators": [20,50,100,200], # Menor que en RF para evitar sobreajuste
                          
                          "max_depth": [1,2,3,4,5], 
                          
                          "subsample": [1.0, 0.8, 0.5], 
                          
                          "max_features": ["sqrt", 3, 4], 
                          }

gs_gradient_boosting = GridSearchCV(gradient_boosting,
                                    grid_gradient_boosting,
                                    cv=10,
                                    scoring="accuracy",
                                    verbose=1,
                                    n_jobs=-1)
```


```python
gs_gradient_boosting.fit(df_train[nombres_features], df_train["color"])
```

    Fitting 10 folds for each of 540 candidates, totalling 5400 fits
    

    [Parallel(n_jobs=-1)]: Using backend LokyBackend with 8 concurrent workers.
    [Parallel(n_jobs=-1)]: Done  52 tasks      | elapsed:    1.0s
    [Parallel(n_jobs=-1)]: Done 307 tasks      | elapsed:   11.0s
    [Parallel(n_jobs=-1)]: Done 557 tasks      | elapsed:   23.1s
    [Parallel(n_jobs=-1)]: Done 907 tasks      | elapsed:   43.6s
    [Parallel(n_jobs=-1)]: Done 1357 tasks      | elapsed:  1.3min
    [Parallel(n_jobs=-1)]: Done 1907 tasks      | elapsed:  2.3min
    [Parallel(n_jobs=-1)]: Done 2557 tasks      | elapsed:  2.8min
    [Parallel(n_jobs=-1)]: Done 3307 tasks      | elapsed:  3.8min
    [Parallel(n_jobs=-1)]: Done 4157 tasks      | elapsed:  4.8min
    [Parallel(n_jobs=-1)]: Done 5107 tasks      | elapsed:  5.9min
    [Parallel(n_jobs=-1)]: Done 5400 out of 5400 | elapsed:  6.3min finished
    




    GridSearchCV(cv=10, error_score='raise-deprecating',
                 estimator=GradientBoostingClassifier(criterion='friedman_mse',
                                                      init=None, learning_rate=0.1,
                                                      loss='deviance', max_depth=3,
                                                      max_features=None,
                                                      max_leaf_nodes=None,
                                                      min_impurity_decrease=0.0,
                                                      min_impurity_split=None,
                                                      min_samples_leaf=1,
                                                      min_samples_split=2,
                                                      min_weight_fraction_leaf=0.0,
                                                      n_estimators=100,
                                                      n_iter_n...
                                                      subsample=1.0, tol=0.0001,
                                                      validation_fraction=0.1,
                                                      verbose=0, warm_start=False),
                 iid='warn', n_jobs=-1,
                 param_grid={'learning_rate': [0.05, 0.1, 0.5],
                             'loss': ['deviance'], 'max_depth': [1, 2, 3, 4, 5],
                             'max_features': ['sqrt', 3, 4],
                             'n_estimators': [20, 50, 100, 200],
                             'subsample': [1.0, 0.8, 0.5]},
                 pre_dispatch='2*n_jobs', refit=True, return_train_score=False,
                 scoring='accuracy', verbose=1)




```python
gs_gradient_boosting.best_params_
```




    {'learning_rate': 0.1,
     'loss': 'deviance',
     'max_depth': 5,
     'max_features': 3,
     'n_estimators': 200,
     'subsample': 0.5}




```python
gs_gradient_boosting.best_score_
```




    0.9971137194535309



## 7.- PERCEPTRÓN MULTICAPA

Probamos redes neuronales:


```python
from sklearn.neural_network import MLPClassifier

mlp = Pipeline(steps=[("scaler",StandardScaler()),
                      ("mlp",MLPClassifier())
                     ])

grid_mlp = {"mlp__hidden_layer_sizes": [(4,),            
                                        (4,4),          
                                        (30,),           
                                        (30,30),
                                        (100,),
                                        (100,100,100)],
            "mlp__activation": ["logistic","relu","tanh"], 
            "mlp__solver": ["adam"], 
            "mlp__alpha": [0.0, 0.0001, 0.1], 
            "mlp__validation_fraction": [0.1],            
            "mlp__early_stopping": [True],
            "mlp__max_iter": [6000],
            "mlp__learning_rate_init": [0.001, 0.1, 0.5] 
           }

gs_mlp = GridSearchCV(mlp,
                      grid_mlp,
                      cv=10,
                      scoring="accuracy",
                      verbose=1,
                      n_jobs=-1)
```


```python
gs_mlp.fit(df_train[nombres_features], df_train["color"])
```

    Fitting 10 folds for each of 162 candidates, totalling 1620 fits
    

    [Parallel(n_jobs=-1)]: Using backend LokyBackend with 8 concurrent workers.
    [Parallel(n_jobs=-1)]: Done  34 tasks      | elapsed:    6.0s
    [Parallel(n_jobs=-1)]: Done 184 tasks      | elapsed:   31.4s
    [Parallel(n_jobs=-1)]: Done 434 tasks      | elapsed:  1.1min
    [Parallel(n_jobs=-1)]: Done 784 tasks      | elapsed:  2.2min
    [Parallel(n_jobs=-1)]: Done 1234 tasks      | elapsed:  3.5min
    [Parallel(n_jobs=-1)]: Done 1620 out of 1620 | elapsed:  5.0min finished
    




    GridSearchCV(cv=10, error_score='raise-deprecating',
                 estimator=Pipeline(memory=None,
                                    steps=[('scaler',
                                            StandardScaler(copy=True,
                                                           with_mean=True,
                                                           with_std=True)),
                                           ('mlp',
                                            MLPClassifier(activation='relu',
                                                          alpha=0.0001,
                                                          batch_size='auto',
                                                          beta_1=0.9, beta_2=0.999,
                                                          early_stopping=False,
                                                          epsilon=1e-08,
                                                          hidden_layer_sizes=(100,),
                                                          learning_rate='constant',
                                                          learning_rate_i...
                 param_grid={'mlp__activation': ['logistic', 'relu', 'tanh'],
                             'mlp__alpha': [0.0, 0.0001, 0.1],
                             'mlp__early_stopping': [True],
                             'mlp__hidden_layer_sizes': [(4,), (4, 4), (30,),
                                                         (30, 30), (100,),
                                                         (100, 100, 100)],
                             'mlp__learning_rate_init': [0.001, 0.1, 0.5],
                             'mlp__max_iter': [6000], 'mlp__solver': ['adam'],
                             'mlp__validation_fraction': [0.1]},
                 pre_dispatch='2*n_jobs', refit=True, return_train_score=False,
                 scoring='accuracy', verbose=1)




```python
gs_mlp.best_params_
```




    {'mlp__activation': 'relu',
     'mlp__alpha': 0.0,
     'mlp__early_stopping': True,
     'mlp__hidden_layer_sizes': (100,),
     'mlp__learning_rate_init': 0.1,
     'mlp__max_iter': 6000,
     'mlp__solver': 'adam',
     'mlp__validation_fraction': 0.1}




```python
gs_mlp.best_score_
```




    0.9973061381566288



## Solución

Después de haber probado todos los modelos, tenemos estos resultados:
1.	Regresión Logística: 	0.9975465346907603
2.	Árboles de decisión:	0.9829919206849101	
3.	K Nearest Neighbors:	0.9974152475229384	
4.	Naïve Bayes:		    0.9719068693477005
5.	Naïve Bayes K-best: 	0.9557436982874735  
6.	Perceptrón multicapa:	0.9973061381566288

Probamos 2 Ensembles:	
7.	Random forest:			0.998896036709122		 
8.	Gradient Boosting:	    0.9971137194535309

El mejor modelo es:
**Random Forest de 250 árboles y profundidad 23, con 0.998896036709122**

## Probamos el modelo seleccionado con el conjunto test


```python
mejor_modelo = gs_rf.best_estimator_

mejor_modelo
```




    Pipeline(memory=None,
             steps=[('bosque',
                     RandomForestClassifier(bootstrap=True, class_weight=None,
                                            criterion='gini', max_depth=23,
                                            max_features='auto',
                                            max_leaf_nodes=None,
                                            min_impurity_decrease=0.0,
                                            min_impurity_split=None,
                                            min_samples_leaf=1, min_samples_split=2,
                                            min_weight_fraction_leaf=0.0,
                                            n_estimators=250, n_jobs=None,
                                            oob_score=False, random_state=None,
                                            verbose=0, warm_start=False))],
             verbose=False)




```python
mejor_modelo.fit(df_train[nombres_features], df_train["color"])
```




    Pipeline(memory=None,
             steps=[('bosque',
                     RandomForestClassifier(bootstrap=True, class_weight=None,
                                            criterion='gini', max_depth=23,
                                            max_features='auto',
                                            max_leaf_nodes=None,
                                            min_impurity_decrease=0.0,
                                            min_impurity_split=None,
                                            min_samples_leaf=1, min_samples_split=2,
                                            min_weight_fraction_leaf=0.0,
                                            n_estimators=250, n_jobs=None,
                                            oob_score=False, random_state=None,
                                            verbose=0, warm_start=False))],
             verbose=False)




```python
predicciones_test = mejor_modelo.predict(df_test[nombres_features])

predicciones_test
# variables que me escupe el modelo
```




    array([1, 0, 1, ..., 1, 1, 0])




```python
df_test["predicciones"] = predicciones_test
df_test
```

    C:\Users\hesca\Anaconda3\lib\site-packages\ipykernel_launcher.py:1: SettingWithCopyWarning: 
    A value is trying to be set on a copy of a slice from a DataFrame.
    Try using .loc[row_indexer,col_indexer] = value instead
    
    See the caveats in the documentation: http://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
      """Entry point for launching an IPython kernel.
    




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>fixed acidity</th>
      <th>volatile acidity</th>
      <th>citric acid</th>
      <th>residual sugar</th>
      <th>chlorides</th>
      <th>free sulfur dioxide</th>
      <th>total sulfur dioxide</th>
      <th>density</th>
      <th>pH</th>
      <th>sulphates</th>
      <th>alcohol</th>
      <th>quality</th>
      <th>color</th>
      <th>predicciones</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>2598</td>
      <td>6.8</td>
      <td>0.210</td>
      <td>0.36</td>
      <td>18.10</td>
      <td>0.046</td>
      <td>32.0</td>
      <td>133.0</td>
      <td>1.00000</td>
      <td>3.27</td>
      <td>0.48</td>
      <td>8.8</td>
      <td>5</td>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <td>590</td>
      <td>7.5</td>
      <td>0.705</td>
      <td>0.24</td>
      <td>1.80</td>
      <td>0.360</td>
      <td>15.0</td>
      <td>63.0</td>
      <td>0.99640</td>
      <td>3.00</td>
      <td>1.59</td>
      <td>9.5</td>
      <td>5</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <td>6234</td>
      <td>7.9</td>
      <td>0.440</td>
      <td>0.26</td>
      <td>4.45</td>
      <td>0.033</td>
      <td>23.0</td>
      <td>100.0</td>
      <td>0.99117</td>
      <td>3.17</td>
      <td>0.52</td>
      <td>12.7</td>
      <td>6</td>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <td>2737</td>
      <td>7.3</td>
      <td>0.400</td>
      <td>0.30</td>
      <td>1.70</td>
      <td>0.080</td>
      <td>33.0</td>
      <td>79.0</td>
      <td>0.99690</td>
      <td>3.41</td>
      <td>0.65</td>
      <td>9.5</td>
      <td>6</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <td>2802</td>
      <td>6.8</td>
      <td>0.360</td>
      <td>0.24</td>
      <td>4.60</td>
      <td>0.039</td>
      <td>24.0</td>
      <td>124.0</td>
      <td>0.99090</td>
      <td>3.27</td>
      <td>0.34</td>
      <td>12.6</td>
      <td>7</td>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <td>6218</td>
      <td>5.1</td>
      <td>0.420</td>
      <td>0.01</td>
      <td>1.50</td>
      <td>0.017</td>
      <td>25.0</td>
      <td>102.0</td>
      <td>0.98940</td>
      <td>3.38</td>
      <td>0.36</td>
      <td>12.3</td>
      <td>7</td>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <td>6280</td>
      <td>6.4</td>
      <td>0.145</td>
      <td>0.49</td>
      <td>5.40</td>
      <td>0.048</td>
      <td>54.0</td>
      <td>164.0</td>
      <td>0.99460</td>
      <td>3.56</td>
      <td>0.44</td>
      <td>10.8</td>
      <td>6</td>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <td>1419</td>
      <td>6.7</td>
      <td>0.130</td>
      <td>0.45</td>
      <td>4.20</td>
      <td>0.043</td>
      <td>52.0</td>
      <td>131.0</td>
      <td>0.99162</td>
      <td>3.06</td>
      <td>0.54</td>
      <td>11.3</td>
      <td>6</td>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <td>3330</td>
      <td>7.0</td>
      <td>0.280</td>
      <td>0.26</td>
      <td>1.70</td>
      <td>0.042</td>
      <td>34.0</td>
      <td>130.0</td>
      <td>0.99250</td>
      <td>3.43</td>
      <td>0.50</td>
      <td>10.7</td>
      <td>8</td>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <td>1260</td>
      <td>7.2</td>
      <td>0.630</td>
      <td>0.03</td>
      <td>2.20</td>
      <td>0.080</td>
      <td>17.0</td>
      <td>88.0</td>
      <td>0.99745</td>
      <td>3.53</td>
      <td>0.58</td>
      <td>9.8</td>
      <td>6</td>
      <td>0</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
<p>1300 rows × 14 columns</p>
</div>



**Vemos el acierto con distintas métricas:**

### AUC-ROC


```python
from sklearn.metrics import roc_auc_score, confusion_matrix, classification_report, f1_score

Score_auc = roc_auc_score(y_true = df_test["color"], y_score = df_test["predicciones"])

print(Score_auc)
# Vemos el auc_score con el test
```

    0.9837850633211032
    

### F1-Score


```python
F1_score = f1_score(y_true = df_test["color"], y_pred = df_test["predicciones"])

print(F1_score)
# Vemos el F1_score con el test
```

    0.9939819458375125
    

### Matriz de Confusión


```python
matriz_confusion = confusion_matrix(y_true = df_test["color"],
                                    y_pred = df_test["predicciones"])
                                
matriz_confusion
```




    array([[297,   9],
           [  3, 991]], dtype=int64)




```python
matriz_confusion_df = pd.DataFrame(matriz_confusion)
label = ['red', 'white']
       
matriz_confusion_df.columns= label
matriz_confusion_df.index = label

# Y nombramos lo que son las columnas y las filas:
matriz_confusion_df.columns.name = "Predicho"
matriz_confusion_df.index.name = "Real"
```


```python
matriz_confusion_df
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th>Predicho</th>
      <th>red</th>
      <th>white</th>
    </tr>
    <tr>
      <th>Real</th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>red</td>
      <td>297</td>
      <td>9</td>
    </tr>
    <tr>
      <td>white</td>
      <td>3</td>
      <td>991</td>
    </tr>
  </tbody>
</table>
</div>




```python
plt.figure(figsize=(8,4))
sns.heatmap(matriz_confusion_df,                     
            annot=True,
            fmt="d",
            cmap="Blues")
pass
```

{% include figure image_path="/images/MachineLearning/output_87_0.png" %}



# B. MODELO DE REGRESIÓN


```python
# Vamos a reutilizar los set de train y test que hemos creado, pero eliminando la varibale color:
df_train_reg = df_train.drop(["color"], axis = 1)
df_train_reg.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>fixed acidity</th>
      <th>volatile acidity</th>
      <th>citric acid</th>
      <th>residual sugar</th>
      <th>chlorides</th>
      <th>free sulfur dioxide</th>
      <th>total sulfur dioxide</th>
      <th>density</th>
      <th>pH</th>
      <th>sulphates</th>
      <th>alcohol</th>
      <th>quality</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>2406</td>
      <td>6.0</td>
      <td>0.22</td>
      <td>0.23</td>
      <td>5.0</td>
      <td>0.045</td>
      <td>10.0</td>
      <td>122.0</td>
      <td>0.99261</td>
      <td>2.94</td>
      <td>0.63</td>
      <td>10.0</td>
      <td>6</td>
    </tr>
    <tr>
      <td>4847</td>
      <td>10.2</td>
      <td>0.36</td>
      <td>0.64</td>
      <td>2.9</td>
      <td>0.122</td>
      <td>10.0</td>
      <td>41.0</td>
      <td>0.99800</td>
      <td>3.23</td>
      <td>0.66</td>
      <td>12.5</td>
      <td>6</td>
    </tr>
    <tr>
      <td>6107</td>
      <td>7.3</td>
      <td>0.26</td>
      <td>0.31</td>
      <td>1.6</td>
      <td>0.040</td>
      <td>39.0</td>
      <td>173.0</td>
      <td>0.99180</td>
      <td>3.19</td>
      <td>0.51</td>
      <td>11.4</td>
      <td>6</td>
    </tr>
    <tr>
      <td>2082</td>
      <td>6.6</td>
      <td>0.50</td>
      <td>0.26</td>
      <td>11.3</td>
      <td>0.029</td>
      <td>32.0</td>
      <td>110.0</td>
      <td>0.99302</td>
      <td>3.27</td>
      <td>0.78</td>
      <td>12.9</td>
      <td>8</td>
    </tr>
    <tr>
      <td>2601</td>
      <td>6.3</td>
      <td>0.40</td>
      <td>0.24</td>
      <td>5.1</td>
      <td>0.036</td>
      <td>43.0</td>
      <td>131.0</td>
      <td>0.99186</td>
      <td>3.24</td>
      <td>0.44</td>
      <td>11.3</td>
      <td>6</td>
    </tr>
  </tbody>
</table>
</div>




```python
df_test_reg = df_test.drop(["color"], axis = 1)
```


```python
# Creamos listas con todas las columnas del dataset, y de las features del problema de regresión:
nombres_col_reg = ["fixed acidity","volatile acidity","citric acid","residual sugar","chlorides","free sulfur dioxide",
                    "total sulfur dioxide","density","pH","sulphates","alcohol","quality"]
nombres_fea_reg = ["fixed acidity","volatile acidity","citric acid","residual sugar","chlorides","free sulfur dioxide",
                    "total sulfur dioxide","density","pH","sulphates","alcohol"]
```


```python
# Vamos a ver cuantos observaciones hay de cada nota
df['quality'].value_counts() 
```




    6    2836
    5    2138
    7    1079
    4     216
    8     193
    3      30
    9       5
    Name: quality, dtype: int64



Observamos que la mayoría se concentran entre los valores 5 y 7


```python
# Para este modelo vamos a realizar todos los Pipelines de golpe, además, vamos a probar a incrustar dentro 
# de nuestro pipeline en algunos modelos un seleccionador de variables (aunque como dijimos antes no sería necesario 
# solo con 11 features):

from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import RFECV, SelectKBest, f_regression

# 1.-REGRESIÓN LINEAL
from sklearn.linear_model import LinearRegression
linreg_rfecv = Pipeline(steps=[("scaler", StandardScaler()),
                               ("rfecv",RFECV(estimator=LinearRegression())),
                               ("linreg", LinearRegression())
                              ])

# 2.-ÁRBOL DECISIÓN REGRESSOR
from sklearn.tree import DecisionTreeRegressor
arbol_reg = DecisionTreeRegressor

# 3.-RANDOM FOREST REGRESSOR (probamos ensemble bagging)
from sklearn.ensemble import RandomForestRegressor
rf_reg = RandomForestRegressor()   # Lo mismo que en el anterior

# 4.-GRADIENT BOOSTING (probamos ensemble boosting)
from sklearn.ensemble import GradientBoostingRegressor
gb_reg = GradientBoostingRegressor()   

# 5.-K-NEAREST NEIGHBORS REGRESSOR
from sklearn.neighbors import KNeighborsRegressor
neigh_reg = Pipeline([("scaler",StandardScaler()),
                      ("rfecv",RFECV(estimator=LinearRegression())),
                      ("knr", KNeighborsRegressor())
                     ])

# 6.-SUPPORT VECTOR REGRESSOR
from sklearn.svm import SVR
svr_reg = Pipeline([("scaler",StandardScaler()),
                    ("rfecv",RFECV(estimator=LinearRegression())),
                    ("svr",SVR())
                     ])

```


```python
# Ahora vamos a seleccionar los hiperparámetros que vamos a probar:
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# 1.-REGRESIÓN LINEAL
grid_linreg_rfecv = {"rfecv__step": [1], # Probamos a quitar features de una en una.
                     "rfecv__cv": [5],
                    }

# 2.-ÁRBOL DECISIÓN REGRESSOR
grid_arbol_reg = {"criterion": ["mae"],  # probamos con mae
                  "max_depth": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]          
                  }

# 3.-RANDOM FOREST REGRESSOR 
grid_rf_reg = {"n_estimators": [200,500],                      
               "max_depth": [3,5,10,15,20,25,30, 35, 50], 
               "max_features": ["sqrt", 3, 4] 
              }


# 4.-GRADIENT BOOSTING 
grid_gb_reg = {"learning_rate": [0.05, 0.1, 0.5], 
               "n_estimators": [20,50,100,200],
               "max_depth": [1,2,3,4,5],
               "subsample": [1.0, 0.8, 0.5],
               "max_features": ["sqrt", 3, 4], 
               }   

# 5.-K-NEAREST NEIGHBORS REGRESSOR
grid_neigh_reg = {"rfecv__step": [1], 
                  "rfecv__cv": [5,10],
                  "knr__n_neighbors": [3,5,7,9,11,13,15,17,19], 
                  "knr__weights": ["uniform","distance"]}

# 6.-SUPPORT VECTOR REGRESSOR
grid_svr_reg = {"rfecv__step": [1],
                "rfecv__cv": [5],
                "svr__C": [0.01, 0.1, 0.5, 1.0],
                "svr__degree": [2,3,4],
                "svr__gamma": [0.001, 0.1, "auto", 1.0]
                    }

```


```python
# El tercer paso es construir el GridsearchCV con cada pipeline y su grid de hiperparámetros.
# Decidimos utilzar MAE ya que consideramos que una diferencia de calidad de 2 puntos es el doble que 1, 
# y con esta métrica podemos representar esto de manera más adecuada.

from sklearn.metrics import mean_absolute_error

# 1.-REGRESIÓN LINEAL
gs_linreg_rfecv = GridSearchCV(linreg_rfecv,
                               grid_linreg_rfecv,
                               cv=10,
                               scoring="neg_mean_absolute_error",
                               verbose=1,
                               n_jobs=-1)

# 2.-ÁRBOL DECISIÓN REGRESSOR
gs_arbol_reg = GridSearchCV(arbol_reg,
                            grid_arbol_reg,
                            cv=10,
                            scoring="neg_mean_absolute_error",
                            verbose=1,
                            n_jobs=-1)

# 3.-RANDOM FOREST REGRESSOR 
gs_rf_reg = GridSearchCV(rf_reg,
                        grid_rf_reg,
                        cv=10,
                        scoring="neg_mean_absolute_error",
                        verbose=1,
                        n_jobs=-1)

# 4.-GRADIENT BOOSTING  
gs_gradient_boosting = GridSearchCV(gb_reg,
                                    grid_gb_reg,
                                    cv=10,
                                    scoring="neg_mean_absolute_error",
                                    verbose=1,
                                    n_jobs=-1)

# 5.-K-NEAREST NEIGHBORS REGRESSOR
gs_neigh_reg = GridSearchCV(neigh_reg,
                                  grid_neigh_reg,
                                  cv=10,
                                  scoring="neg_mean_absolute_error",
                                  verbose=1,
                                  n_jobs=-1)

# 6.-SUPPORT VECTOR REGRESSOR
gs_svr = GridSearchCV(svr_reg,
                      grid_svr_reg,
                      cv=10,
                      scoring="neg_mean_absolute_error",
                      verbose=1,
                      n_jobs=-1)

```

**AHORA COMENZAMOS A PROBAR MODELOS**


```python
# 1.-REGRESIÓN LINEAL
gs_linreg_rfecv.fit(df_train_reg[nombres_fea_reg], df_train_reg["quality"])
```

    Fitting 10 folds for each of 1 candidates, totalling 10 fits
    

    [Parallel(n_jobs=-1)]: Using backend LokyBackend with 8 concurrent workers.
    [Parallel(n_jobs=-1)]: Done   6 out of  10 | elapsed:    0.2s remaining:    0.1s
    [Parallel(n_jobs=-1)]: Done  10 out of  10 | elapsed:    0.3s finished
    




    GridSearchCV(cv=10, error_score='raise-deprecating',
                 estimator=Pipeline(memory=None,
                                    steps=[('scaler',
                                            StandardScaler(copy=True,
                                                           with_mean=True,
                                                           with_std=True)),
                                           ('rfecv',
                                            RFECV(cv='warn',
                                                  estimator=LinearRegression(copy_X=True,
                                                                             fit_intercept=True,
                                                                             n_jobs=None,
                                                                             normalize=False),
                                                  min_features_to_select=1,
                                                  n_jobs=None, scoring=None, step=1,
                                                  verbose=0)),
                                           ('linreg',
                                            LinearRegression(copy_X=True,
                                                             fit_intercept=True,
                                                             n_jobs=None,
                                                             normalize=False))],
                                    verbose=False),
                 iid='warn', n_jobs=-1,
                 param_grid={'rfecv__cv': [5], 'rfecv__step': [1]},
                 pre_dispatch='2*n_jobs', refit=True, return_train_score=False,
                 scoring='neg_mean_absolute_error', verbose=1)




```python
gs_linreg_rfecv.best_params_
```




    {'rfecv__cv': 5, 'rfecv__step': 1}




```python
gs_linreg_rfecv.best_score_
```




    -0.5667713603697664




```python
# 2.-ÁRBOL DECISIÓN REGRESSOR

gs_arbol_reg.fit(df_train_reg[nombres_fea_reg], df_train_reg["quality"])
```

    Fitting 10 folds for each of 10 candidates, totalling 100 fits
    

    [Parallel(n_jobs=-1)]: Using backend LokyBackend with 8 concurrent workers.
    [Parallel(n_jobs=-1)]: Done  34 tasks      | elapsed:    7.3s
    [Parallel(n_jobs=-1)]: Done 100 out of 100 | elapsed:   34.8s finished
    




    GridSearchCV(cv=10, error_score='raise-deprecating',
                 estimator=DecisionTreeRegressor(criterion='mse', max_depth=None,
                                                 max_features=None,
                                                 max_leaf_nodes=None,
                                                 min_impurity_decrease=0.0,
                                                 min_impurity_split=None,
                                                 min_samples_leaf=1,
                                                 min_samples_split=2,
                                                 min_weight_fraction_leaf=0.0,
                                                 presort=False, random_state=None,
                                                 splitter='best'),
                 iid='warn', n_jobs=-1,
                 param_grid={'criterion': ['mae'],
                             'max_depth': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]},
                 pre_dispatch='2*n_jobs', refit=True, return_train_score=False,
                 scoring='neg_mean_absolute_error', verbose=1)




```python
gs_arbol_reg.best_params_
```




    {'criterion': 'mae', 'max_depth': 9}




```python
gs_arbol_reg.best_score_
```




    -0.5114489128343275




```python
# 3.-RANDOM FOREST REGRESSOR 
gs_rf_reg.fit(df_train_reg[nombres_fea_reg], df_train_reg["quality"])
```

    Fitting 10 folds for each of 54 candidates, totalling 540 fits
    

    [Parallel(n_jobs=-1)]: Using backend LokyBackend with 8 concurrent workers.
    [Parallel(n_jobs=-1)]: Done  34 tasks      | elapsed:   11.4s
    [Parallel(n_jobs=-1)]: Done 184 tasks      | elapsed:  1.6min
    [Parallel(n_jobs=-1)]: Done 434 tasks      | elapsed:  6.3min
    [Parallel(n_jobs=-1)]: Done 540 out of 540 | elapsed:  8.3min finished
    




    GridSearchCV(cv=10, error_score='raise-deprecating',
                 estimator=RandomForestRegressor(bootstrap=True, criterion='mse',
                                                 max_depth=None,
                                                 max_features='auto',
                                                 max_leaf_nodes=None,
                                                 min_impurity_decrease=0.0,
                                                 min_impurity_split=None,
                                                 min_samples_leaf=1,
                                                 min_samples_split=2,
                                                 min_weight_fraction_leaf=0.0,
                                                 n_estimators='warn', n_jobs=None,
                                                 oob_score=False, random_state=None,
                                                 verbose=0, warm_start=False),
                 iid='warn', n_jobs=-1,
                 param_grid={'max_depth': [3, 5, 10, 15, 20, 25, 30, 35, 50],
                             'max_features': ['sqrt', 3, 4],
                             'n_estimators': [200, 500]},
                 pre_dispatch='2*n_jobs', refit=True, return_train_score=False,
                 scoring='neg_mean_absolute_error', verbose=1)




```python
gs_rf_reg.best_params_
```




    {'max_depth': 25, 'max_features': 3, 'n_estimators': 500}




```python
gs_rf_reg.best_score_
```




    -0.4277938679627371




```python
# 4.-GRADIENT BOOSTING  
gs_gradient_boosting.fit(df_train_reg[nombres_fea_reg], df_train_reg["quality"])
```

    Fitting 10 folds for each of 540 candidates, totalling 5400 fits
    

    [Parallel(n_jobs=-1)]: Using backend LokyBackend with 8 concurrent workers.
    [Parallel(n_jobs=-1)]: Done  34 tasks      | elapsed:    4.1s
    [Parallel(n_jobs=-1)]: Done 184 tasks      | elapsed:    7.9s
    [Parallel(n_jobs=-1)]: Done 578 tasks      | elapsed:   20.4s
    [Parallel(n_jobs=-1)]: Done 1066 tasks      | elapsed:   41.4s
    [Parallel(n_jobs=-1)]: Done 1516 tasks      | elapsed:  1.1min
    [Parallel(n_jobs=-1)]: Done 2066 tasks      | elapsed:  1.7min
    [Parallel(n_jobs=-1)]: Done 2716 tasks      | elapsed:  2.0min
    [Parallel(n_jobs=-1)]: Done 3466 tasks      | elapsed:  2.8min
    [Parallel(n_jobs=-1)]: Done 4655 tasks      | elapsed:  3.6min
    [Parallel(n_jobs=-1)]: Done 5400 out of 5400 | elapsed:  4.5min finished
    




    GridSearchCV(cv=10, error_score='raise-deprecating',
                 estimator=GradientBoostingRegressor(alpha=0.9,
                                                     criterion='friedman_mse',
                                                     init=None, learning_rate=0.1,
                                                     loss='ls', max_depth=3,
                                                     max_features=None,
                                                     max_leaf_nodes=None,
                                                     min_impurity_decrease=0.0,
                                                     min_impurity_split=None,
                                                     min_samples_leaf=1,
                                                     min_samples_split=2,
                                                     min_weight_fraction_leaf=0.0,
                                                     n_estimators=100,
                                                     n_ite...
                                                     subsample=1.0, tol=0.0001,
                                                     validation_fraction=0.1,
                                                     verbose=0, warm_start=False),
                 iid='warn', n_jobs=-1,
                 param_grid={'learning_rate': [0.05, 0.1, 0.5],
                             'max_depth': [1, 2, 3, 4, 5],
                             'max_features': ['sqrt', 3, 4],
                             'n_estimators': [20, 50, 100, 200],
                             'subsample': [1.0, 0.8, 0.5]},
                 pre_dispatch='2*n_jobs', refit=True, return_train_score=False,
                 scoring='neg_mean_absolute_error', verbose=1)




```python
gs_gradient_boosting.best_params_
```




    {'learning_rate': 0.5,
     'max_depth': 5,
     'max_features': 'sqrt',
     'n_estimators': 200,
     'subsample': 1.0}




```python
gs_gradient_boosting.best_score_
```




    -0.46954907702548876




```python
# 5.-K-NEAREST NEIGHBORS REGRESSOR
gs_neigh_reg.fit(df_train_reg[nombres_fea_reg], df_train_reg["quality"])
```

    Fitting 10 folds for each of 36 candidates, totalling 360 fits
    

    [Parallel(n_jobs=-1)]: Using backend LokyBackend with 8 concurrent workers.
    [Parallel(n_jobs=-1)]: Done  34 tasks      | elapsed:    5.8s
    [Parallel(n_jobs=-1)]: Done 184 tasks      | elapsed:   14.3s
    [Parallel(n_jobs=-1)]: Done 360 out of 360 | elapsed:   24.8s finished
    




    GridSearchCV(cv=10, error_score='raise-deprecating',
                 estimator=Pipeline(memory=None,
                                    steps=[('scaler',
                                            StandardScaler(copy=True,
                                                           with_mean=True,
                                                           with_std=True)),
                                           ('rfecv',
                                            RFECV(cv='warn',
                                                  estimator=LinearRegression(copy_X=True,
                                                                             fit_intercept=True,
                                                                             n_jobs=None,
                                                                             normalize=False),
                                                  min_features_to_select=1,
                                                  n_jobs=None, scoring=None, step=1,
                                                  verbose=0)),
                                           ('knr',
                                            KNeighb...
                                                                metric='minkowski',
                                                                metric_params=None,
                                                                n_jobs=None,
                                                                n_neighbors=5, p=2,
                                                                weights='uniform'))],
                                    verbose=False),
                 iid='warn', n_jobs=-1,
                 param_grid={'knr__n_neighbors': [3, 5, 7, 9, 11, 13, 15, 17, 19],
                             'knr__weights': ['uniform', 'distance'],
                             'rfecv__cv': [5, 10], 'rfecv__step': [1]},
                 pre_dispatch='2*n_jobs', refit=True, return_train_score=False,
                 scoring='neg_mean_absolute_error', verbose=1)




```python
gs_neigh_reg.best_params_
```




    {'knr__n_neighbors': 15,
     'knr__weights': 'distance',
     'rfecv__cv': 5,
     'rfecv__step': 1}




```python
gs_neigh_reg.best_score_
```




    -0.4052324726227335




```python
# 6.-SUPPORT VECTOR REGRESSOR
gs_svr.fit(df_train_reg[nombres_fea_reg], df_train_reg["quality"])
```

    Fitting 10 folds for each of 48 candidates, totalling 480 fits
    

    [Parallel(n_jobs=-1)]: Using backend LokyBackend with 8 concurrent workers.
    [Parallel(n_jobs=-1)]: Done  34 tasks      | elapsed:   23.3s
    [Parallel(n_jobs=-1)]: Done 184 tasks      | elapsed:  1.6min
    [Parallel(n_jobs=-1)]: Done 434 tasks      | elapsed:  4.1min
    [Parallel(n_jobs=-1)]: Done 480 out of 480 | elapsed:  4.6min finished
    




    GridSearchCV(cv=10, error_score='raise-deprecating',
                 estimator=Pipeline(memory=None,
                                    steps=[('scaler',
                                            StandardScaler(copy=True,
                                                           with_mean=True,
                                                           with_std=True)),
                                           ('rfecv',
                                            RFECV(cv='warn',
                                                  estimator=LinearRegression(copy_X=True,
                                                                             fit_intercept=True,
                                                                             n_jobs=None,
                                                                             normalize=False),
                                                  min_features_to_select=1,
                                                  n_jobs=None, scoring=None, step=1,
                                                  verbose=0)),
                                           ('svr',
                                            SVR(C=1...
                                                gamma='auto_deprecated',
                                                kernel='rbf', max_iter=-1,
                                                shrinking=True, tol=0.001,
                                                verbose=False))],
                                    verbose=False),
                 iid='warn', n_jobs=-1,
                 param_grid={'rfecv__cv': [5], 'rfecv__step': [1],
                             'svr__C': [0.01, 0.1, 0.5, 1.0],
                             'svr__degree': [2, 3, 4],
                             'svr__gamma': [0.001, 0.1, 'auto', 1.0]},
                 pre_dispatch='2*n_jobs', refit=True, return_train_score=False,
                 scoring='neg_mean_absolute_error', verbose=1)




```python
gs_svr.best_params_
```




    {'rfecv__cv': 5,
     'rfecv__step': 1,
     'svr__C': 1.0,
     'svr__degree': 2,
     'svr__gamma': 1.0}




```python
gs_svr.best_score_
```




    -0.46826146219157583



**LOS RESULTADOS HAN SIDO:**
1.	 Regresión Lineal: 		-0.5667713603697664
2.	Árbol decisión Regressor:	-0.5114489128343275
3.	Random Forest Regressor:	-0.4277938679627371
4.	Gradient Boosting:		-0.46954907702548876
5.	K-Nearest Neighbors Regressor	-0.4052324726227335
6.	Support Vector Regressor:	-0.46826146219157583

Por lo tanto, y según el Valor del Score Mean Absolute Error, el modelo que minimiza el Error es K-Nearest Neighbors Regressor con -0.4052324726227335, y tiene los siguientes parámetros:

{'knr__n_neighbors': 15,
 'knr__weights': 'distance',
 'rfecv__cv': 5,
 'rfecv__step': 1}


### PROBAMOS EL MEJOR MODELO


```python
from sklearn.model_selection import GridSearchCV # para definir los hiperparámetros que va a utilizar el pipeline.
```


```python
mejor_modelo_reg = gs_neigh_reg.best_estimator_

mejor_modelo_reg
```




    Pipeline(memory=None,
             steps=[('scaler',
                     StandardScaler(copy=True, with_mean=True, with_std=True)),
                    ('rfecv',
                     RFECV(cv=5,
                           estimator=LinearRegression(copy_X=True,
                                                      fit_intercept=True,
                                                      n_jobs=None,
                                                      normalize=False),
                           min_features_to_select=1, n_jobs=None, scoring=None,
                           step=1, verbose=0)),
                    ('knr',
                     KNeighborsRegressor(algorithm='auto', leaf_size=30,
                                         metric='minkowski', metric_params=None,
                                         n_jobs=None, n_neighbors=15, p=2,
                                         weights='distance'))],
             verbose=False)




```python
mejor_modelo_reg.fit(df_train_reg[nombres_fea_reg], df_train_reg["quality"])
```




    Pipeline(memory=None,
             steps=[('scaler',
                     StandardScaler(copy=True, with_mean=True, with_std=True)),
                    ('rfecv',
                     RFECV(cv=5,
                           estimator=LinearRegression(copy_X=True,
                                                      fit_intercept=True,
                                                      n_jobs=None,
                                                      normalize=False),
                           min_features_to_select=1, n_jobs=None, scoring=None,
                           step=1, verbose=0)),
                    ('knr',
                     KNeighborsRegressor(algorithm='auto', leaf_size=30,
                                         metric='minkowski', metric_params=None,
                                         n_jobs=None, n_neighbors=15, p=2,
                                         weights='distance'))],
             verbose=False)




```python
predicciones_test_reg = mejor_modelo_reg.predict(df_test[nombres_fea_reg])

predicciones_test_reg
```




    array([5.        , 5.40468598, 6.92516879, ..., 6.3588261 , 6.51893453,
           5.18474551])




```python
df_test_reg["predicciones"]=predicciones_test_reg

df_test_reg
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>fixed acidity</th>
      <th>volatile acidity</th>
      <th>citric acid</th>
      <th>residual sugar</th>
      <th>chlorides</th>
      <th>free sulfur dioxide</th>
      <th>total sulfur dioxide</th>
      <th>density</th>
      <th>pH</th>
      <th>sulphates</th>
      <th>alcohol</th>
      <th>quality</th>
      <th>predicciones</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>2598</td>
      <td>6.8</td>
      <td>0.210</td>
      <td>0.36</td>
      <td>18.10</td>
      <td>0.046</td>
      <td>32.0</td>
      <td>133.0</td>
      <td>1.00000</td>
      <td>3.27</td>
      <td>0.48</td>
      <td>8.8</td>
      <td>5</td>
      <td>5.000000</td>
    </tr>
    <tr>
      <td>590</td>
      <td>7.5</td>
      <td>0.705</td>
      <td>0.24</td>
      <td>1.80</td>
      <td>0.360</td>
      <td>15.0</td>
      <td>63.0</td>
      <td>0.99640</td>
      <td>3.00</td>
      <td>1.59</td>
      <td>9.5</td>
      <td>5</td>
      <td>5.404686</td>
    </tr>
    <tr>
      <td>6234</td>
      <td>7.9</td>
      <td>0.440</td>
      <td>0.26</td>
      <td>4.45</td>
      <td>0.033</td>
      <td>23.0</td>
      <td>100.0</td>
      <td>0.99117</td>
      <td>3.17</td>
      <td>0.52</td>
      <td>12.7</td>
      <td>6</td>
      <td>6.925169</td>
    </tr>
    <tr>
      <td>2737</td>
      <td>7.3</td>
      <td>0.400</td>
      <td>0.30</td>
      <td>1.70</td>
      <td>0.080</td>
      <td>33.0</td>
      <td>79.0</td>
      <td>0.99690</td>
      <td>3.41</td>
      <td>0.65</td>
      <td>9.5</td>
      <td>6</td>
      <td>5.707296</td>
    </tr>
    <tr>
      <td>2802</td>
      <td>6.8</td>
      <td>0.360</td>
      <td>0.24</td>
      <td>4.60</td>
      <td>0.039</td>
      <td>24.0</td>
      <td>124.0</td>
      <td>0.99090</td>
      <td>3.27</td>
      <td>0.34</td>
      <td>12.6</td>
      <td>7</td>
      <td>6.617467</td>
    </tr>
    <tr>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <td>6218</td>
      <td>5.1</td>
      <td>0.420</td>
      <td>0.01</td>
      <td>1.50</td>
      <td>0.017</td>
      <td>25.0</td>
      <td>102.0</td>
      <td>0.98940</td>
      <td>3.38</td>
      <td>0.36</td>
      <td>12.3</td>
      <td>7</td>
      <td>6.592602</td>
    </tr>
    <tr>
      <td>6280</td>
      <td>6.4</td>
      <td>0.145</td>
      <td>0.49</td>
      <td>5.40</td>
      <td>0.048</td>
      <td>54.0</td>
      <td>164.0</td>
      <td>0.99460</td>
      <td>3.56</td>
      <td>0.44</td>
      <td>10.8</td>
      <td>6</td>
      <td>5.996269</td>
    </tr>
    <tr>
      <td>1419</td>
      <td>6.7</td>
      <td>0.130</td>
      <td>0.45</td>
      <td>4.20</td>
      <td>0.043</td>
      <td>52.0</td>
      <td>131.0</td>
      <td>0.99162</td>
      <td>3.06</td>
      <td>0.54</td>
      <td>11.3</td>
      <td>6</td>
      <td>6.358826</td>
    </tr>
    <tr>
      <td>3330</td>
      <td>7.0</td>
      <td>0.280</td>
      <td>0.26</td>
      <td>1.70</td>
      <td>0.042</td>
      <td>34.0</td>
      <td>130.0</td>
      <td>0.99250</td>
      <td>3.43</td>
      <td>0.50</td>
      <td>10.7</td>
      <td>8</td>
      <td>6.518935</td>
    </tr>
    <tr>
      <td>1260</td>
      <td>7.2</td>
      <td>0.630</td>
      <td>0.03</td>
      <td>2.20</td>
      <td>0.080</td>
      <td>17.0</td>
      <td>88.0</td>
      <td>0.99745</td>
      <td>3.53</td>
      <td>0.58</td>
      <td>9.8</td>
      <td>6</td>
      <td>5.184746</td>
    </tr>
  </tbody>
</table>
<p>1300 rows × 13 columns</p>
</div>



## Probamos distintas métricas para valorar el acierto


```python
# Mean Absolute Error
mae = mean_absolute_error(df_test_reg["quality"],df_test_reg["predicciones"])
mae
```




    0.4118528787183134




```python
# Mean Squared Error
mse = mean_squared_error(df_test_reg["quality"],df_test_reg["predicciones"])
mse
```




    0.41041292474968977




```python
# Root Mean Squared Error
from math import sqrt
rmse = sqrt(mean_squared_error(df_test_reg["quality"],df_test_reg["predicciones"]))
rmse
```




    0.6406347826567722




```python
# r2_score
r2 = r2_score(df_test_reg["quality"],df_test_reg["predicciones"])
r2
```




    0.4958086837869017



## Conclusión

Atendiendo a las distintas métricas, vemos que nuestro modelo no es bueno.
Con MSA vemos que el error medio está entorno al 0.4, es decir las desviación promedio que tiene nuestro modelo a la hora de predecir.

Analizando el problema desde la realidad vemos que se intenta predecir la nota que va a tener un vino, algo totalmente subjetivo y que ponen expertos en el tema, con datos de composición química que representan los vinos.

Creo que si en este estudio consiguiéramos un modelo con una muy buena predicción conseguiríamos la fórmula "secreta" para conseguir "buenos" vinos a nivel puntuación, y por lo tanto ventas del mismo. Este estudio demuestra que no hay una única composición química (lo que se traduciría imagino en el tipo de uva seleccionado, la fermentación,... y otros temas que desconozco) para conseguir "buenos" vinos.
