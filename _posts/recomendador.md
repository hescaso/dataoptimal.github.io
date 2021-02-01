---
title: "Sistema de Recomendación"
date: 2021-02-01
tags: [Recommender systems, Sistemas de recomendación, collaborative filtering recommender system, filtrado colaborativo]
header:
  image: "/images/recom.png"
excerpt: "Recommender systems, Sistemas de recomendación, collaborative filtering recommender system, filtrado colaborativo"
classes: "wide"
mathjax: "true"

---


RECOMENDADOR DE PELÍCULAS




Uno de las aplicaciones más utilizadas que ha aportado el Machine Learning son los sistemas de Recomendación. Debido al grado de efectividad que tienen podemos verlos cada día en una multitud de de servicios que consumimos: Netflix recomendando películas y series, Spotify sugiriendo canciones y artistas, o Amazón recomendandote nuevos artículos para comprar. También podemos ver esto en pequeños blogs que nos ofrecen lecturas recomendadas, o los propios periodicos digitales. 

En estos ejemplos se podía intuir con claridad que detrás de las recomendaciones hay un algoritmo funcionando, pero en otras ocasiones no es tan claro. Por ejemplo, Netflix no solo recomienda películas y series, sino que para la mayoría de estas tiene varias carátulas para mostrarte y según el análisis de tu usuario te mostrará una u otra para hacerte más atractiva su recomendación. 

Un ejemplo de lo anterior que me ha pasado recientemente: hace unos meses vi la serie *Gambito de Dama*, la cual protagoniza **Anya Taylor-Joy** haciendo un papel soberbio. Pues desde entonces en la serie *Peaky Blinders* (que no he visto de momento), me aparece la carátula con su cara. Entiendo que ella aparece en esta serie e intentan aprovechar el tirón con esta actriz. 

Los sistemas de recomendación, a veces llamados en inglés “recommender systems” son algoritmos que intentan “predecir” los siguientes ítems (productos, canciones, etc.) que querrá adquirir un usuario en particular.

Antes del Machine Learning, lo más común era usar “rankings” ó listas con lo más votado, ó más popular de entre todos los productos. Entonces a todos los usuarios se les recomendaba lo mismo. Es una técnica que aún se usa y en muchos casos funciona bien, por ejemplo, en librerías ponen apartados con los libros más vendidos, best sellers. 

## Tipos de motores
Estos son algunas de los métodos de recomendación más utilizados:

**Popularity**: Aconseja por la “popularidad” de los productos. Por ejemplo, “los más vendidos” globalmente, se ofrecerán a todos los usuarios por igual sin aprovechar la personalización. Es fácil de implementar y en algunos casos es efectiva. Esto podemos verlo por ejemplo en la *Casa del libro*, donde siempre tenemos al entrar en la tienda el top 10 en ventas.

**Content-based**: A partir de productos visitados por el usuario, se intenta “adivinar” qué busca el usuario y ofrecer mercancías similares. Un ejemplo clásico es Amazón, en la que guiado por nuestras visitas (no hace falta que se compre) nos ofrece productos similares hasta que el algoritmo detecta nuestro cambio de "necesidad".

**Colaborative**: Es el más novedoso, pues utiliza la información de “masas” para identificar perfiles similares y aprender de los datos para recomendar productos de manera individual. Este tipo de sistema de recomendación es el que vamos a ver en detalle.

Se basa en el supuesto de que si las personas coinciden en gustos en el pasado también lo harán en el futuro.
PROS: Fácil de implementar con resultado acertado.
CONTRAS: Sin un ranking inicial no es posible tener una recomendación. 


## Comenzamos a contruir nuestro motor

Existen varias formas de construir un sistema de recomendación *colaborative*.

En nuestro caso vamos a probar varios métodos, en primer lugar correlaremos nuestra matriz con el método de Pearson, y en segundo lugar utilizaremos la librería Surprise (Surprise es una herramienta de Python para construir y analizar sistemas de recomendación que tratan con datos de calificación explícitos) y probaremos algunos de sus algoritmos para ver cual arroja el menor error posible.

***OBTENEMOS LOS DATOS***

Obtenemos el archivo de películas de la página https://grouplens.org/datasets/movielens/. En este caso, y para facilitar los cálculos, utilizaremos el pequeño que tiene 100.000 valoraciones (el completo tiene 27 millones!!!)

Los datos consisten en:
	- 100,000 valoraciones entre 1 y 5 de 943 usuarios sobre 1682 películas clásicas. 
	- Cada usuarios valora al menos 20 películas. 
    - Se completa el archivo con datos demográficos básicos como edad, genéro, ocupación,...


```python
# Importamos pandas
import pandas as pd
import numpy as np

# Tenemos los datos divididos en varios archivos. Nosotros valomos a utilizar en principio los archivos "u.data" 
# con la información del usuario, el id de la película y la valoración, y el archivo "u.item" del que únicamente 
# rescataremos los datos de id de la película y el título.

colum_usu = ['usuario_id', 'pelicula_id', 'valoracion']  
valora_usu = pd.read_csv('C:/Users/hesca/Documents/DataSets/ml-100k/u.data',
                      sep='\t', names=colum_usu, usecols=range(3), encoding="ISO-8859-1")

colum_pelis = ['pelicula_id', 'titulo']  
peliculas = pd.read_csv('C:/Users/hesca/Documents/DataSets/ml-100k/u.item', sep='|',
                     names=colum_pelis, usecols=range(2), encoding="ISO-8859-1")

# Combinamos ambos datasets ...
valoraciones = pd.merge(peliculas, valora_usu)
```


```python
# Ahora voy a votar yo mismo algunas películas, y utilizaré el recomendador para descubrir nuevas películas
colum_vot = ['usuario_id', 'titulo', 'valoracion'] 
misValoraciones = pd.read_csv('C:/Users/hesca/Documents/DataSets/ml-100k/misVotaciones.csv',
                            sep=';', names=colum_vot, usecols=range(3), encoding="ISO-8859-1")

# Estas son las películas que he votado
misValoraciones
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
      <th>usuario_id</th>
      <th>titulo</th>
      <th>valoracion</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>0</td>
      <td>999</td>
      <td>Aladdin (1992)</td>
      <td>3.0</td>
    </tr>
    <tr>
      <td>1</td>
      <td>999</td>
      <td>Braveheart (1995)</td>
      <td>3.0</td>
    </tr>
    <tr>
      <td>2</td>
      <td>999</td>
      <td>Clockwork Orange, A (1971)</td>
      <td>4.0</td>
    </tr>
    <tr>
      <td>3</td>
      <td>999</td>
      <td>Dances with Wolves (1990)</td>
      <td>3.0</td>
    </tr>
    <tr>
      <td>4</td>
      <td>999</td>
      <td>English Patient, The (1996)</td>
      <td>3.0</td>
    </tr>
    <tr>
      <td>5</td>
      <td>999</td>
      <td>Face/Off (1997)</td>
      <td>2.0</td>
    </tr>
    <tr>
      <td>6</td>
      <td>999</td>
      <td>Forrest Gump (1994)</td>
      <td>4.0</td>
    </tr>
    <tr>
      <td>7</td>
      <td>999</td>
      <td>Game, The (1997)</td>
      <td>3.0</td>
    </tr>
    <tr>
      <td>8</td>
      <td>999</td>
      <td>Godfather, The (1972)</td>
      <td>5.0</td>
    </tr>
    <tr>
      <td>9</td>
      <td>999</td>
      <td>Jurassic Park (1993)</td>
      <td>2.5</td>
    </tr>
    <tr>
      <td>10</td>
      <td>999</td>
      <td>Lion King, The (1994)</td>
      <td>2.5</td>
    </tr>
    <tr>
      <td>11</td>
      <td>999</td>
      <td>Pulp Fiction (1994)</td>
      <td>5.0</td>
    </tr>
    <tr>
      <td>12</td>
      <td>999</td>
      <td>Reservoir Dogs (1992)</td>
      <td>4.5</td>
    </tr>
    <tr>
      <td>13</td>
      <td>999</td>
      <td>Return of the Jedi (1983)</td>
      <td>2.0</td>
    </tr>
    <tr>
      <td>14</td>
      <td>999</td>
      <td>Rock, The (1996)</td>
      <td>3.0</td>
    </tr>
    <tr>
      <td>15</td>
      <td>999</td>
      <td>Scream (1996)</td>
      <td>1.0</td>
    </tr>
    <tr>
      <td>16</td>
      <td>999</td>
      <td>Seven (Se7en) (1995)</td>
      <td>4.0</td>
    </tr>
    <tr>
      <td>17</td>
      <td>999</td>
      <td>Silence of the Lambs, The (1991)</td>
      <td>4.0</td>
    </tr>
    <tr>
      <td>18</td>
      <td>999</td>
      <td>Star Wars (1977)</td>
      <td>1.0</td>
    </tr>
    <tr>
      <td>19</td>
      <td>999</td>
      <td>Terminator 2: Judgment Day (1991)</td>
      <td>3.0</td>
    </tr>
    <tr>
      <td>20</td>
      <td>999</td>
      <td>Titanic (1997)</td>
      <td>1.5</td>
    </tr>
    <tr>
      <td>21</td>
      <td>999</td>
      <td>Trainspotting (1996)</td>
      <td>5.0</td>
    </tr>
    <tr>
      <td>22</td>
      <td>999</td>
      <td>Toy Story (1995)</td>
      <td>2.5</td>
    </tr>
    <tr>
      <td>23</td>
      <td>999</td>
      <td>Good Will Hunting (1997)</td>
      <td>5.0</td>
    </tr>
    <tr>
      <td>24</td>
      <td>999</td>
      <td>Schindler's List (1993)</td>
      <td>4.0</td>
    </tr>
    <tr>
      <td>25</td>
      <td>999</td>
      <td>Fargo (1996)</td>
      <td>4.0</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Unimos nuestra votación al total de datos
valoraciones = pd.concat([valoraciones[['titulo','usuario_id','valoracion']], misValoraciones],sort=False, axis=0)
valoraciones
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
      <th>titulo</th>
      <th>usuario_id</th>
      <th>valoracion</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>0</td>
      <td>Toy Story (1995)</td>
      <td>308</td>
      <td>4.0</td>
    </tr>
    <tr>
      <td>1</td>
      <td>Toy Story (1995)</td>
      <td>287</td>
      <td>5.0</td>
    </tr>
    <tr>
      <td>2</td>
      <td>Toy Story (1995)</td>
      <td>148</td>
      <td>4.0</td>
    </tr>
    <tr>
      <td>3</td>
      <td>Toy Story (1995)</td>
      <td>280</td>
      <td>4.0</td>
    </tr>
    <tr>
      <td>4</td>
      <td>Toy Story (1995)</td>
      <td>66</td>
      <td>3.0</td>
    </tr>
    <tr>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <td>21</td>
      <td>Trainspotting (1996)</td>
      <td>999</td>
      <td>5.0</td>
    </tr>
    <tr>
      <td>22</td>
      <td>Toy Story (1995)</td>
      <td>999</td>
      <td>2.5</td>
    </tr>
    <tr>
      <td>23</td>
      <td>Good Will Hunting (1997)</td>
      <td>999</td>
      <td>5.0</td>
    </tr>
    <tr>
      <td>24</td>
      <td>Schindler's List (1993)</td>
      <td>999</td>
      <td>4.0</td>
    </tr>
    <tr>
      <td>25</td>
      <td>Fargo (1996)</td>
      <td>999</td>
      <td>4.0</td>
    </tr>
  </tbody>
</table>
<p>100026 rows × 3 columns</p>
</div>




```python
# Pivotamos la tabla para crear una matriz con una fila por usuario, una columna por película y la votación 
# que se le dio a la misma.  
ValoracionPeliculas = valoraciones.pivot_table(index=['usuario_id'],columns=['titulo'],values='valoracion')  
ValoracionPeliculas
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
      <th>titulo</th>
      <th>'Til There Was You (1997)</th>
      <th>1-900 (1994)</th>
      <th>101 Dalmatians (1996)</th>
      <th>12 Angry Men (1957)</th>
      <th>187 (1997)</th>
      <th>2 Days in the Valley (1996)</th>
      <th>20,000 Leagues Under the Sea (1954)</th>
      <th>2001: A Space Odyssey (1968)</th>
      <th>3 Ninjas: High Noon At Mega Mountain (1998)</th>
      <th>39 Steps, The (1935)</th>
      <th>...</th>
      <th>Yankee Zulu (1994)</th>
      <th>Year of the Horse (1997)</th>
      <th>You So Crazy (1994)</th>
      <th>Young Frankenstein (1974)</th>
      <th>Young Guns (1988)</th>
      <th>Young Guns II (1990)</th>
      <th>Young Poisoner's Handbook, The (1995)</th>
      <th>Zeus and Roxanne (1997)</th>
      <th>unknown</th>
      <th>Á köldum klaka (Cold Fever) (1994)</th>
    </tr>
    <tr>
      <th>usuario_id</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>1</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>2.0</td>
      <td>5.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>3.0</td>
      <td>4.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>5.0</td>
      <td>3.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>4.0</td>
      <td>NaN</td>
    </tr>
    <tr>
      <td>2</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>1.0</td>
      <td>NaN</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <td>3</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>2.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <td>4</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <td>5</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>2.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>4.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>4.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>4.0</td>
      <td>NaN</td>
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
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <td>940</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <td>941</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <td>942</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>3.0</td>
      <td>NaN</td>
      <td>3.0</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <td>943</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>2.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>4.0</td>
      <td>3.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <td>999</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>
<p>944 rows × 1664 columns</p>
</div>



Perfecto, ya tenemos nuestra matriz con la valoracón de cada película que ha puesto cada usuario.
Antes de comenzar con nuestro recomendador, vamos a probar a correlar una película con el resto, según las valoraciones, para ver películas parecidas a esta.


```python
# Vamos a probar con la película Fou Rooms:
FouRoomsValoracion = ValoracionPeliculas['Four Rooms (1995)']

# Correlamos el resto de películas (columnas) con la seleccionada (Four Rooms)  
pelisParecidas = ValoracionPeliculas.corrwith(FouRoomsValoracion)  
pelisParecidas = pelisParecidas.dropna()  
df = pd.DataFrame(pelisParecidas)

# Las ordenamos por el valor de score que hemos generado, de forma descendente  
pelisParecidas.sort_values(ascending=False)
```

    C:\Users\hesca\Anaconda3\lib\site-packages\numpy\lib\function_base.py:2526: RuntimeWarning: Degrees of freedom <= 0 for slice
      c = cov(x, y, rowvar)
    C:\Users\hesca\Anaconda3\lib\site-packages\numpy\lib\function_base.py:2455: RuntimeWarning: divide by zero encountered in true_divide
      c *= np.true_divide(1, fact)
    




    titulo
    Purple Noon (1960)                        1.0
    Roseanna's Grave (For Roseanna) (1997)    1.0
    Man of the House (1995)                   1.0
    Little Princess, A (1995)                 1.0
    Bushwhacked (1995)                        1.0
                                             ... 
    Inspector General, The (1949)            -1.0
    Kissed (1996)                            -1.0
    Man of No Importance, A (1994)           -1.0
    Mark of Zorro, The (1940)                -1.0
    Little Odessa (1994)                     -1.0
    Length: 1137, dtype: float64



Como podemos ver, hay muchas películas con el nivel de parecido máximo (1.0), pero sin embargo, son muy desconocidas. Esto puede deberse a que haya películas con muy pocas valoraciones pero que se de la casualidad de que dos o tres usuarios hayan valorado a Four Rooms y a estas películas con la misma puntuación.
Vamos a comprobar el número de votaciones de Purple Noon por ejemplo:


```python
ValoracionPeliculas['Purple Noon (1960)'].count()
```




    7



En efecto tiene solo 7 valoraciones.

Para solucionar el hecho de que películas poco votadas tengan tanto peso en nuestro recomendador y pierda eficacia, lo que haremos será agregar las votaciones por película para coger solo aquellas películas que tengan al menos 50 valoraciones de usuarios distintos. 


```python
# Agregamos por título y devolvemos el número de veces que se puntuó, y la media de la puntuación  
peliculasVotadas = valoraciones.groupby('titulo').agg({'valoracion': [np.size, np.mean]})

peliculasVotadas

```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead tr th {
        text-align: left;
    }

    .dataframe thead tr:last-of-type th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr>
      <th></th>
      <th colspan="2" halign="left">valoracion</th>
    </tr>
    <tr>
      <th></th>
      <th>size</th>
      <th>mean</th>
    </tr>
    <tr>
      <th>titulo</th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>'Til There Was You (1997)</td>
      <td>9.0</td>
      <td>2.333333</td>
    </tr>
    <tr>
      <td>1-900 (1994)</td>
      <td>5.0</td>
      <td>2.600000</td>
    </tr>
    <tr>
      <td>101 Dalmatians (1996)</td>
      <td>109.0</td>
      <td>2.908257</td>
    </tr>
    <tr>
      <td>12 Angry Men (1957)</td>
      <td>125.0</td>
      <td>4.344000</td>
    </tr>
    <tr>
      <td>187 (1997)</td>
      <td>41.0</td>
      <td>3.024390</td>
    </tr>
    <tr>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <td>Young Guns II (1990)</td>
      <td>44.0</td>
      <td>2.772727</td>
    </tr>
    <tr>
      <td>Young Poisoner's Handbook, The (1995)</td>
      <td>41.0</td>
      <td>3.341463</td>
    </tr>
    <tr>
      <td>Zeus and Roxanne (1997)</td>
      <td>6.0</td>
      <td>2.166667</td>
    </tr>
    <tr>
      <td>unknown</td>
      <td>9.0</td>
      <td>3.444444</td>
    </tr>
    <tr>
      <td>Á köldum klaka (Cold Fever) (1994)</td>
      <td>1.0</td>
      <td>3.000000</td>
    </tr>
  </tbody>
</table>
<p>1664 rows × 2 columns</p>
</div>




```python
# Nos quedamos con todas las que tengan mas de 50 puntuaciones de distintos usuarios  
peliculasPopulares = peliculasVotadas['valoracion']['size'] >= 50
```


```python
# Ordenamos por la puntuación asignada  
peliculasVotadas[peliculasPopulares].sort_values([('valoracion', 'mean')], ascending=False)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead tr th {
        text-align: left;
    }

    .dataframe thead tr:last-of-type th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr>
      <th></th>
      <th colspan="2" halign="left">valoracion</th>
    </tr>
    <tr>
      <th></th>
      <th>size</th>
      <th>mean</th>
    </tr>
    <tr>
      <th>titulo</th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>Close Shave, A (1995)</td>
      <td>112.0</td>
      <td>4.491071</td>
    </tr>
    <tr>
      <td>Wrong Trousers, The (1993)</td>
      <td>118.0</td>
      <td>4.466102</td>
    </tr>
    <tr>
      <td>Schindler's List (1993)</td>
      <td>299.0</td>
      <td>4.464883</td>
    </tr>
    <tr>
      <td>Casablanca (1942)</td>
      <td>243.0</td>
      <td>4.456790</td>
    </tr>
    <tr>
      <td>Wallace &amp; Gromit: The Best of Aardman Animation (1996)</td>
      <td>67.0</td>
      <td>4.447761</td>
    </tr>
    <tr>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <td>Cable Guy, The (1996)</td>
      <td>106.0</td>
      <td>2.339623</td>
    </tr>
    <tr>
      <td>Beautician and the Beast, The (1997)</td>
      <td>86.0</td>
      <td>2.313953</td>
    </tr>
    <tr>
      <td>Striptease (1996)</td>
      <td>67.0</td>
      <td>2.238806</td>
    </tr>
    <tr>
      <td>McHale's Navy (1997)</td>
      <td>69.0</td>
      <td>2.188406</td>
    </tr>
    <tr>
      <td>Island of Dr. Moreau, The (1996)</td>
      <td>57.0</td>
      <td>2.157895</td>
    </tr>
  </tbody>
</table>
<p>605 rows × 2 columns</p>
</div>



Podemos ver todas aquellas películas que tienen más de 50 valoraciones de distintos usuarios, ordenadas por su puntuación media. Si ahora hacemos un “join”; con la tabla de valoraciones original, nos quedaremos solo con estas peliculas, descartando aquellas que solo valoraron unos pocos usuarios:


```python
# Hacemos el join  
df = peliculasVotadas[peliculasPopulares].join(pd.DataFrame(pelisParecidas, columns=['similitud']))

# Ordenamos el dataframe por similitud, y vemos los primeros 10 resultados  
df.sort_values(['similitud'], ascending=False)[:10]  

```

    C:\Users\hesca\Anaconda3\lib\site-packages\pandas\core\reshape\merge.py:617: UserWarning: merging between different levels can give an unintended result (2 levels on the left, 1 on the right)
      warnings.warn(msg, UserWarning)
    




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
      <th>(valoracion, size)</th>
      <th>(valoracion, mean)</th>
      <th>similitud</th>
    </tr>
    <tr>
      <th>titulo</th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>Apostle, The (1997)</td>
      <td>55.0</td>
      <td>3.654545</td>
      <td>1.000000</td>
    </tr>
    <tr>
      <td>Four Rooms (1995)</td>
      <td>90.0</td>
      <td>3.033333</td>
      <td>1.000000</td>
    </tr>
    <tr>
      <td>Notorious (1946)</td>
      <td>52.0</td>
      <td>4.115385</td>
      <td>0.870388</td>
    </tr>
    <tr>
      <td>Philadelphia Story, The (1940)</td>
      <td>104.0</td>
      <td>4.115385</td>
      <td>0.834726</td>
    </tr>
    <tr>
      <td>Excess Baggage (1997)</td>
      <td>52.0</td>
      <td>2.538462</td>
      <td>0.771744</td>
    </tr>
    <tr>
      <td>Mrs. Brown (Her Majesty, Mrs. Brown) (1997)</td>
      <td>96.0</td>
      <td>3.947917</td>
      <td>0.743161</td>
    </tr>
    <tr>
      <td>Mary Shelley's Frankenstein (1994)</td>
      <td>59.0</td>
      <td>3.067797</td>
      <td>0.729866</td>
    </tr>
    <tr>
      <td>Seven Years in Tibet (1997)</td>
      <td>155.0</td>
      <td>3.458065</td>
      <td>0.724176</td>
    </tr>
    <tr>
      <td>Life Less Ordinary, A (1997)</td>
      <td>53.0</td>
      <td>3.075472</td>
      <td>0.704779</td>
    </tr>
    <tr>
      <td>Nosferatu (Nosferatu, eine Symphonie des Grauens) (1922)</td>
      <td>54.0</td>
      <td>3.555556</td>
      <td>0.690066</td>
    </tr>
  </tbody>
</table>
</div>



Vemos que tenemos 2 películas con similitud perfecta, la propia *Four Rooms*, y *The Apostle*.Con esto hemos comprobado cómo podemos encontrar películas similares a la propuesta según las votaciones de los distintos usuarios. Pasamos a construir nuestro modelo.

***CONSTRUYENDO EL MOTOR DE RECOMENDACIÓN POR CORRELACIÓN***

Ahora que hemos visto un ejemplo de como encontrar similitudes entre pelÍculas, podemos avanzar y tratar de generar recomendaciones para un usuario basadas en su actividad anterior (en su histórico de puntuaciones). Es muy parecido a lo que hemos hecho hasta ahora… esta vez lo que haremos será, en lugar de correlar una película con las demás, correlar todas con todas, del siguiente modo:


```python
# Correlamos todas las columnas con todas las demás usando el metodo pearson habiendo descartado todas aquellas
# que no tengan al menos 50 valoraciones de usuarios  
corrMatrix = ValoracionPeliculas.corr(method='pearson', min_periods=50)  
corrMatrix.head(20) 
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
      <th>titulo</th>
      <th>'Til There Was You (1997)</th>
      <th>1-900 (1994)</th>
      <th>101 Dalmatians (1996)</th>
      <th>12 Angry Men (1957)</th>
      <th>187 (1997)</th>
      <th>2 Days in the Valley (1996)</th>
      <th>20,000 Leagues Under the Sea (1954)</th>
      <th>2001: A Space Odyssey (1968)</th>
      <th>3 Ninjas: High Noon At Mega Mountain (1998)</th>
      <th>39 Steps, The (1935)</th>
      <th>...</th>
      <th>Yankee Zulu (1994)</th>
      <th>Year of the Horse (1997)</th>
      <th>You So Crazy (1994)</th>
      <th>Young Frankenstein (1974)</th>
      <th>Young Guns (1988)</th>
      <th>Young Guns II (1990)</th>
      <th>Young Poisoner's Handbook, The (1995)</th>
      <th>Zeus and Roxanne (1997)</th>
      <th>unknown</th>
      <th>Á köldum klaka (Cold Fever) (1994)</th>
    </tr>
    <tr>
      <th>titulo</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>'Til There Was You (1997)</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <td>1-900 (1994)</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <td>101 Dalmatians (1996)</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>1.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <td>12 Angry Men (1957)</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>1.000000</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0.178848</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0.096546</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <td>187 (1997)</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <td>2 Days in the Valley (1996)</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>1.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <td>20,000 Leagues Under the Sea (1954)</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>1.000000</td>
      <td>0.259308</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <td>2001: A Space Odyssey (1968)</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0.178848</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0.259308</td>
      <td>1.000000</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>-0.001307</td>
      <td>-0.174918</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <td>3 Ninjas: High Noon At Mega Mountain (1998)</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <td>39 Steps, The (1935)</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>1.0</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <td>8 1/2 (1963)</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <td>8 Heads in a Duffel Bag (1997)</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <td>8 Seconds (1994)</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <td>A Chef in Love (1996)</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <td>Above the Rim (1994)</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <td>Absolute Power (1997)</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <td>Abyss, The (1989)</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0.089206</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0.140161</td>
      <td>0.384703</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <td>Ace Ventura: Pet Detective (1994)</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0.138417</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0.221837</td>
      <td>0.360457</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <td>Ace Ventura: When Nature Calls (1995)</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <td>Across the Sea of Time (1995)</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>
<p>20 rows × 1664 columns</p>
</div>



En esta tabla podemos ver la correlación entre unas películas y otras. Vemos que hay gran cantidad de Nan, esto es debido a que no se cumple la condición de que haya 50 valoraciones para alguna de las películas que forman la celda de nuestra matriz. A estas matrices que tienen tantos valores nulos se las denomina matrices "sparse" o "dispersas".

Ahora sí, vamos a utilizar nuestro usuario (id=999), y según sus votaciones utilizaremos nuestro recomendador. 


```python
# Seleccionamos el usuario 999 y eliminamos todas las columnas que tengan nulo (películas no vistas)  
misValoraciones = ValoracionPeliculas.loc[999].dropna()

# Recordamos las pelítuclas valoradas  
misValoraciones
```




    titulo
    Aladdin (1992)                       3.0
    Braveheart (1995)                    3.0
    Clockwork Orange, A (1971)           4.0
    Dances with Wolves (1990)            3.0
    English Patient, The (1996)          3.0
    Face/Off (1997)                      2.0
    Fargo (1996)                         4.0
    Forrest Gump (1994)                  4.0
    Game, The (1997)                     3.0
    Godfather, The (1972)                5.0
    Good Will Hunting (1997)             5.0
    Jurassic Park (1993)                 2.5
    Lion King, The (1994)                2.5
    Pulp Fiction (1994)                  5.0
    Reservoir Dogs (1992)                4.5
    Return of the Jedi (1983)            2.0
    Rock, The (1996)                     3.0
    Schindler's List (1993)              4.0
    Scream (1996)                        1.0
    Seven (Se7en) (1995)                 4.0
    Silence of the Lambs, The (1991)     4.0
    Star Wars (1977)                     1.0
    Terminator 2: Judgment Day (1991)    3.0
    Titanic (1997)                       1.5
    Toy Story (1995)                     2.5
    Trainspotting (1996)                 5.0
    Name: 999, dtype: float64



Hemos valorado 25 películas. Con esta información y correlando estas películas con el resto vamos a comprobar cómo funciona nuestro recomendador.


```python
posiblesSimilares = pd.Series()

# Creamos un bucle para recorrer cada película que he votado
for i in range(0, len(misValoraciones.index)):  
    
# Obtenemos el grado de similitud de las películas bajo la premisa de haber sido puntuadas más de 50 veces. 
    similares = corrMatrix[misValoraciones.index[i]].dropna()

# Multiplicamos el score de correlación por la puntuación asignada por el usuario  
    similares = similares.map(lambda x: x * misValoraciones[i])

# Añadimos la película y la nueva puntuación a nuestra lista de candidatas  
    posiblesSimilares = posiblesSimilares.append(similares)

# Agrupamos los resultados, ya que si una película es muy parecida a varias de las que ha visto el usuario, aparecerá 
# varias veces. En este caso vamos a sumar la "nueva puntuación" cada vez que sale, ya que si aparece muchas veces 
# sumará más puntos y será muy recomendable, y entonces saldrá de las primeras.
posiblesSimilares = posiblesSimilares.groupby(posiblesSimilares.index).sum()

# Finalmente eliminamos todas las peliculas que el usuario ya habia valorado para no incluirlas en la recomendación,  
# le decimos que ignore errores para evitar excepciones si hay problemas con los titulos  
filtered = posiblesSimilares.drop(misValoraciones.index,errors='ignore')  

# Vemos las 5 películas "más" recomendadas
filtered.sort_values(ascending=False).head(10)
```




    Cape Fear (1991)                          20.098064
    Field of Dreams (1989)                    18.909795
    Shawshank Redemption, The (1994)          18.618572
    Kingpin (1996)                            18.456364
    Long Kiss Goodnight, The (1996)           17.624929
    One Flew Over the Cuckoo's Nest (1975)    17.584799
    River Wild, The (1994)                    16.658393
    Die Hard (1988)                           16.609444
    Shine (1996)                              16.515340
    Stand by Me (1986)                        16.499208
    dtype: float64



Estas películas son las que me recomienda, la verdad es que no he visto casi ninguna así que tendré que verlas para saber el grado de efectividad del recomendador :-)

***LIBRERIA SUPRISE***

Ahora vamos a utilizar la librería Surprise (Podéis encontrar más información en http://surpriselib.com/)

Esta librería es bastante completa y muy especializada en este tipo de tarea. De una manera simple vamos a construir nuestro recomendador con los algoritmos: 
- NMF: Un algoritmo de filtrado colaborativo basado en factorización matricial no negativa.
- SVD: El famoso algoritmo SVD, popularizado por Simon Funk durante el Premio Netflix. Equivalente a la factorización matricial probabilística. 
- SVD++: Una mejora del algoritmo SVD que tiene en cuenta las valoraciones implícitas.
- KNN with Z-Score: Un algoritmo de filtrado colaborativo básico que tiene en cuenta una calificación de referencia.
- Co-Clustering: Un algoritmo de filtrado colaborativo basado en la agrupación conjunta.

Hemos visto que cada algoritmo recomendaba películas distintas. Podemos evaluar estos algoritmos dividiendo nuestro conjunto de datos en train y test y medir el rendimiento en el conjunto de datos de prueba. Aplicaremos Cross Validation (k-fold of k = 3) y obtendremos el RMSE promedio.



```python
# Utilizamos nuevamente las películas con más de 50 valoraciones
valoraciones['n_votaciones'] = valoraciones.groupby(['titulo'])['valoracion'].transform('count')
valoraciones= valoraciones[valoraciones.n_votaciones>50][['usuario_id', 'titulo', 'valoracion']]
```


```python
# Importamos la librería Surprise y sus métodos
from surprise import NMF, SVD, SVDpp, KNNBasic, KNNWithMeans, KNNWithZScore, CoClustering
from surprise.model_selection import cross_validate
from surprise import Reader, Dataset
```

Instanciamos la clase Reader y la utilizamos para colocar los datos según el orden necesario. Esta clase es usada para analizar los datos del archivo. Se supone que el archivo especifica una calificación por línea, y cada línea debe respetar la siguiente estructura: **| user | item | rating | (timestamp) |** donde el timestamp es opcional.


```python
algo = Reader(rating_scale=(1, 5))
datos = Dataset.load_from_df(valoraciones, algo)
```

Ahora eliminamos del dataset las películas que hemos visto.


```python
# Obtenemos la lista de películas
listaPeliculas = valoraciones['titulo'].unique()
# Las películas votadas
misVotaciones = valoraciones.loc[valoraciones['usuario_id']==999, 'titulo']
# Eliminamos nuestras películas
peliculas_predecir = np.setdiff1d(listaPeliculas,misVotaciones)
```

**Creamos el recomendador con el algoritmo NMF**


```python
nmf = NMF()
nmf.fit(datos.build_full_trainset())
my_recs = []
for iid in peliculas_predecir:
    my_recs.append((iid, nmf.predict(uid=8,iid=iid).est))
    
pd.DataFrame(my_recs, columns=['titulo', 'prediccion']).sort_values('prediccion', ascending=False).head(10)
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
      <th>titulo</th>
      <th>prediccion</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>115</td>
      <td>Close Shave, A (1995)</td>
      <td>5.000000</td>
    </tr>
    <tr>
      <td>96</td>
      <td>Casablanca (1942)</td>
      <td>5.000000</td>
    </tr>
    <tr>
      <td>75</td>
      <td>Boot, Das (1981)</td>
      <td>4.956953</td>
    </tr>
    <tr>
      <td>551</td>
      <td>Wallace &amp; Gromit: The Best of Aardman Animatio...</td>
      <td>4.904733</td>
    </tr>
    <tr>
      <td>245</td>
      <td>High Noon (1952)</td>
      <td>4.860578</td>
    </tr>
    <tr>
      <td>545</td>
      <td>Vertigo (1958)</td>
      <td>4.858045</td>
    </tr>
    <tr>
      <td>1</td>
      <td>12 Angry Men (1957)</td>
      <td>4.808425</td>
    </tr>
    <tr>
      <td>513</td>
      <td>Thin Man, The (1934)</td>
      <td>4.790323</td>
    </tr>
    <tr>
      <td>567</td>
      <td>Wrong Trousers, The (1993)</td>
      <td>4.782324</td>
    </tr>
    <tr>
      <td>164</td>
      <td>Dr. Strangelove or: How I Learned to Stop Worr...</td>
      <td>4.776931</td>
    </tr>
  </tbody>
</table>
</div>



**Creamos el recomendador con el algoritmo SVD**


```python
svd = SVD()
svd.fit(datos.build_full_trainset())
my_recs = []
for iid in peliculas_predecir:
    my_recs.append((iid, svd.predict(uid=8,iid=iid).est))
    
pd.DataFrame(my_recs, columns=['titulo', 'prediccion']).sort_values('prediccion', ascending=False).head(10)
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
      <th>titulo</th>
      <th>prediccion</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>115</td>
      <td>Close Shave, A (1995)</td>
      <td>4.948758</td>
    </tr>
    <tr>
      <td>461</td>
      <td>Shawshank Redemption, The (1994)</td>
      <td>4.907975</td>
    </tr>
    <tr>
      <td>300</td>
      <td>Lawrence of Arabia (1962)</td>
      <td>4.850240</td>
    </tr>
    <tr>
      <td>383</td>
      <td>One Flew Over the Cuckoo's Nest (1975)</td>
      <td>4.806374</td>
    </tr>
    <tr>
      <td>75</td>
      <td>Boot, Das (1981)</td>
      <td>4.769054</td>
    </tr>
    <tr>
      <td>416</td>
      <td>Raise the Red Lantern (1991)</td>
      <td>4.767982</td>
    </tr>
    <tr>
      <td>418</td>
      <td>Ran (1985)</td>
      <td>4.732029</td>
    </tr>
    <tr>
      <td>252</td>
      <td>Hoop Dreams (1994)</td>
      <td>4.731066</td>
    </tr>
    <tr>
      <td>422</td>
      <td>Rear Window (1954)</td>
      <td>4.729110</td>
    </tr>
    <tr>
      <td>567</td>
      <td>Wrong Trousers, The (1993)</td>
      <td>4.721393</td>
    </tr>
  </tbody>
</table>
</div>



**Creamos el recomendador con el algoritmo SVD++**


```python
svdpp = SVDpp()
svdpp.fit(datos.build_full_trainset())
my_recs = []
for iid in peliculas_predecir:
    my_recs.append((iid, svdpp.predict(uid=8,iid=iid).est))
    
pd.DataFrame(my_recs, columns=['titulo', 'prediccion']).sort_values('prediccion', ascending=False).head(10)
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
      <th>titulo</th>
      <th>prediccion</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>383</td>
      <td>One Flew Over the Cuckoo's Nest (1975)</td>
      <td>4.967189</td>
    </tr>
    <tr>
      <td>1</td>
      <td>12 Angry Men (1957)</td>
      <td>4.944020</td>
    </tr>
    <tr>
      <td>300</td>
      <td>Lawrence of Arabia (1962)</td>
      <td>4.934206</td>
    </tr>
    <tr>
      <td>294</td>
      <td>L.A. Confidential (1997)</td>
      <td>4.918136</td>
    </tr>
    <tr>
      <td>96</td>
      <td>Casablanca (1942)</td>
      <td>4.897174</td>
    </tr>
    <tr>
      <td>525</td>
      <td>To Kill a Mockingbird (1962)</td>
      <td>4.879085</td>
    </tr>
    <tr>
      <td>461</td>
      <td>Shawshank Redemption, The (1994)</td>
      <td>4.872577</td>
    </tr>
    <tr>
      <td>75</td>
      <td>Boot, Das (1981)</td>
      <td>4.850577</td>
    </tr>
    <tr>
      <td>375</td>
      <td>North by Northwest (1959)</td>
      <td>4.829080</td>
    </tr>
    <tr>
      <td>22</td>
      <td>Amadeus (1984)</td>
      <td>4.819654</td>
    </tr>
  </tbody>
</table>
</div>



**Creamos el recomendador con el algoritmo KNN with Z-Score**


```python
KNN = KNNWithZScore()
KNN.fit(datos.build_full_trainset())
my_recs = []
for iid in peliculas_predecir:
    my_recs.append((iid, KNN.predict(uid=8,iid=iid).est))
    
pd.DataFrame(my_recs, columns=['titulo', 'prediccion']).sort_values('prediccion', ascending=False).head(10)
```

    Computing the msd similarity matrix...
    Done computing similarity matrix.
    




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
      <th>titulo</th>
      <th>prediccion</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>567</td>
      <td>Wrong Trousers, The (1993)</td>
      <td>4.994374</td>
    </tr>
    <tr>
      <td>115</td>
      <td>Close Shave, A (1995)</td>
      <td>4.931717</td>
    </tr>
    <tr>
      <td>461</td>
      <td>Shawshank Redemption, The (1994)</td>
      <td>4.923373</td>
    </tr>
    <tr>
      <td>414</td>
      <td>Raiders of the Lost Ark (1981)</td>
      <td>4.886318</td>
    </tr>
    <tr>
      <td>96</td>
      <td>Casablanca (1942)</td>
      <td>4.865282</td>
    </tr>
    <tr>
      <td>300</td>
      <td>Lawrence of Arabia (1962)</td>
      <td>4.836160</td>
    </tr>
    <tr>
      <td>175</td>
      <td>Empire Strikes Back, The (1980)</td>
      <td>4.831657</td>
    </tr>
    <tr>
      <td>41</td>
      <td>As Good As It Gets (1997)</td>
      <td>4.818604</td>
    </tr>
    <tr>
      <td>383</td>
      <td>One Flew Over the Cuckoo's Nest (1975)</td>
      <td>4.792024</td>
    </tr>
    <tr>
      <td>294</td>
      <td>L.A. Confidential (1997)</td>
      <td>4.781272</td>
    </tr>
  </tbody>
</table>
</div>



**Creamos el recomendador con el algoritmo Co-Clustering**


```python
clust = CoClustering()
clust.fit(datos.build_full_trainset())
my_recs = []
for iid in peliculas_predecir:
    my_recs.append((iid, clust.predict(uid=8,iid=iid).est))
    
pd.DataFrame(my_recs, columns=['titulo', 'prediccion']).sort_values('prediccion', ascending=False).head(10)
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
      <th>titulo</th>
      <th>prediccion</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>115</td>
      <td>Close Shave, A (1995)</td>
      <td>4.835049</td>
    </tr>
    <tr>
      <td>567</td>
      <td>Wrong Trousers, The (1993)</td>
      <td>4.810079</td>
    </tr>
    <tr>
      <td>96</td>
      <td>Casablanca (1942)</td>
      <td>4.800768</td>
    </tr>
    <tr>
      <td>551</td>
      <td>Wallace &amp; Gromit: The Best of Aardman Animatio...</td>
      <td>4.791739</td>
    </tr>
    <tr>
      <td>461</td>
      <td>Shawshank Redemption, The (1994)</td>
      <td>4.789207</td>
    </tr>
    <tr>
      <td>422</td>
      <td>Rear Window (1954)</td>
      <td>4.731538</td>
    </tr>
    <tr>
      <td>543</td>
      <td>Usual Suspects, The (1995)</td>
      <td>4.729746</td>
    </tr>
    <tr>
      <td>1</td>
      <td>12 Angry Men (1957)</td>
      <td>4.687978</td>
    </tr>
    <tr>
      <td>108</td>
      <td>Citizen Kane (1941)</td>
      <td>4.681791</td>
    </tr>
    <tr>
      <td>515</td>
      <td>Third Man, The (1949)</td>
      <td>4.677311</td>
    </tr>
  </tbody>
</table>
</div>



**Ahora pasamos a evaluar los diferentes algoritmos** 


```python
cv = []
# Iteramos sobre cada algoritmo
for recsys in [NMF(), SVD(), SVDpp(), KNNWithZScore(), CoClustering()]:
    # Utilizamos cross-validation
    tmp = cross_validate(recsys, datos, measures=['RMSE'], cv=3, verbose=False)
    cv.append((str(recsys).split(' ')[0].split('.')[-1], tmp['test_rmse'].mean()))
pd.DataFrame(cv, columns=['Algoritmo', 'RMSE'])
```

    Computing the msd similarity matrix...
    Done computing similarity matrix.
    Computing the msd similarity matrix...
    Done computing similarity matrix.
    Computing the msd similarity matrix...
    Done computing similarity matrix.
    




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
      <th>Algoritmo</th>
      <th>RMSE</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>0</td>
      <td>NMF</td>
      <td>0.953770</td>
    </tr>
    <tr>
      <td>1</td>
      <td>SVD</td>
      <td>0.929137</td>
    </tr>
    <tr>
      <td>2</td>
      <td>SVDpp</td>
      <td>0.913126</td>
    </tr>
    <tr>
      <td>3</td>
      <td>KNNWithZScore</td>
      <td>0.935942</td>
    </tr>
    <tr>
      <td>4</td>
      <td>CoClustering</td>
      <td>0.945715</td>
    </tr>
  </tbody>
</table>
</div>



**CONCLUSIÓN**

Hemos creado varios sistemas de recomendación de filtrado colaborativo donde teniendo en cuenta el usuario y la película hemos conseguido que el RMSE sea menor que 1. Esto a priori es un buen trabajo, aunque no hay que perder de vista que el RMSE es tan solo una métrica matemática, y que si nos fijamos en el negocio probablemente no siempre queramos minimizar el error ya que esto no siempre será lo óptimo.

Por ejemplo, cuando Spotify nos recomienda canciones si entre ellas hay alguna canción que ya conozcamos y que nos encante esto nos dará confianza en que la selección está bien realizada y que probablemente el resto de canciones también nos pueden gustar.

Por el contrario imaginemos que creamos un sistema que tiene el menor error posible, pero solo nos recomienda cosas nuevas y desconocidas. Si al escuchar las primeras canciones vemos que no nos gustan demasiado pensaremos en que las recomendaciones no sean acertadas, aunque quizá sean las mejores canciones para nosotros y solo nos hace falta escucharlas un par de veces más.

Otro problema que puede surgir es que encierres al usuario en un estilo musical completo. Por ejemplo, si las primeras canciones que escucho fueron todas de rock de los 90´s, puede ser que solo le ofrezcas canciones de rock, o canciones de esa época y se puedan perder canciones muy buenas que no encajen en ese género o década.

Con esto podemos ver que no siempre minimizar el error es lo óptimo, ya que también tiene sus inconvenientes. En este caso si por ejemplo introdujeramos temas populares de otras décadas, aumentaríamos el error, pero probablemente el usuario descubriría más canciones y aumentaría su satisfacción con nuestro motor de recomendación.

Por otro lado, y como complemento a nuestro motor de recomendación por filtrado colaborativo, podemos aplicar otros  algoritmos teniendo en cuenta atributos como el género de la película, la fecha de estreno, el director, el actor, el presupuesto, la duración, etc. 

En este caso, nos referimos a los recomendadores basados en contenido que tratan la recomendación como un problema de clasificación específico del usuario y aprenden un clasificador para los gustos y disgustos del usuario en función de las características de un elemento. En este sistema, las palabras clave se utilizan para describir los elementos y se crea un perfil de usuario para indicar el tipo de elemento que le gusta a este usuario. Por último, incluso podemos tener en cuenta los atributos del usuario, como sexo, edad, ubicación, idioma, etc.

Con una mezcla de ambos algoritmos probablemente conseguiríamos resultados más completos para nuestro algoritmo.

Por último indicar que hay departamentos enteros dedicados a mejorar el sistema de recomendación de su empresa. Este artículo solo pretende acercar de una manera simple la creación de estos modelos.
