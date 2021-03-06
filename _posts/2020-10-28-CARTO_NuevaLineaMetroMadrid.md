---
title: "CARTO: Nueva línea Metro Madrid"
date: 2020-10-28
tags: [CARTO, Visualization, Viz, Geoespacial, Json]
header:
  image: "/images/MapaMadrid.jpg"
excerpt: "CARTO, Visualization, Viz, Geoespacial, Json"
classes: "wide"
mathjax: "true"

---

# Objetivo:

El Consorcio de Transportes de la Comunidad de Madrid (CTM) nos ha encargado un estudio para la posible creación de una nueva línea de metro que agilice la movilidad en el Municipio de Madrid.

# Estudio previo del proyecto:

Para dar respuesta al encargo recibido, se decide analizar el tránsito de viajeros que tiene cada una de las estaciones actualmente, y en función de las estaciones con mayor uso poder crear una ruta que agilice la movilidad.
En primer lugar, tenemos que recopilar los datos necesarios para el análisis.
Tras esto, analizaremos los datos para tener una foto del tránsito de viajeros anualmente.
Realizaremos un análisis visual del estudio mediante la aplicación CARTO.
Por último, emitiremos un informe de conclusiones para dar respuesta a la solicitud.


# Desarrollo del proyecto

(Podemos encontrar la visualización en https://hescaso.carto.com/builder/1467c245-7792-4143-900c-357e7508c99b)


## 1.	Recopilación de datos

Para comenzar buscamos datos de la red de Metro de Madrid que nos pueda ayudar en nuestro análisis. La búsqueda se realiza a través del Portal de Datos Abiertos del Consorcio de Transportes de la Comunidad de Madrid (https://data-crtm.opendata.arcgis.com/), y de la página Metro Madrid que ofrece datos estadísticos de tránsito de viajeros por estación.

Los datasets seleccionados para nuestro análisis son los siguientes:

### Entradas y Utilizaciones por Estaciones
Obtenido a través de la página de Metro de Madrid. Este dataset como su nombre indica, nos da información de las Entradas y Utilizaciones. Definiendo estos conceptos (según nos ha facilitado el propio CTM):
- Entradas: El término “Entradas” se refiere al número de entradas por las barreras de peaje de los vestíbulos de cada estación.
- Utilizaciones: El término “Utilizaciones” se refiere al número de movimientos por el interior de una estación. Para su cálculo se contabilizan las entradas y las salidas por las barreras de peaje de los vestíbulos y se añaden los cambios entre líneas (transbordos) si se trata de estación múltiple con acceso a más de una línea de Metro.

Este dataset nos da la siguiente información:
-	Estaciones de Metro.
-	Línea de Metro correspondiente a cada Estación.
-	Entradas y Utilizaciones de los años 2014, 2015, 2016, 2017 y 2018.

### M4_Estaciones
Obtenido a través del Portal de Datos Abiertos del CTM. Este dataset geolocalizado nos da la siguiente información:
-	ID Estación
-	Código Estación.
-	Denominación
-	Situación
-	Código Provincia
-	Código Municipio
-	Código Vía
-	Nombre Vía
-	Distrito
-	Barrio
-	Latitud
-	Longitud
-	Accesibilidad
-	Situación Calle

### M4_Tramos
Obtenido a través del Portal de Datos Abiertos del CTM. Este dataset geolocalizado nos da la siguiente información útil:
-	Id tramo.
-	Código del itinerario.
-	Código de la línea.
-	Número línea usuario.
-	Sentido.
-	Tipo de itinerario.
-	Código de la Estación.
-	Denominación.
-	Código provincia.
-	Código municipio.
-	Municipio.
-	Dirección.


## 2.	Análisis de los datos

Tras la obtención de los datasets que vamos a utilizar en nuestro análisis, procedemos a revisar los datos para poder introducir las tablas en CARTO.

El primer dataset que vamos a introducir es M4_Paradas, el cual nos va a proporcionar visualmente la situación de las Estaciones sobre el mapa de Madrid.
	
Como posee demasiada información no útil para nuestro análisis, decidimos eliminar las columnas que no vamos a utilizar, dejando únicamente la información relativa a:
-	Estaciones de Metro.
-	Línea de Metro correspondiente a cada Estación.
-	Latitud y Longitud de cada Estación.

Revisamos que se ha cargado correctamente y procedemos a dar estilo a nuestros datos. Cambiamos el color de los puntos 	por el valor línea, utilizando el color real de cada línea de metroque tiene cada estación. Nos encontramos con el primer problema de restricción de colores, ya que solo podemos utilizar 11 colores, pero las líneas reales son 13. Dejamos en otros las Estaciones pertenecientes a las líneas 8, 11, ramales y Metro Ligero.

Tras esto, fijamos el tamaño del punto en 12, y el borde en 2. 

El siguiente dataset incluido es M4_Tramos, como hemos dicho anteriormente es un dataset geolocalizado tipo Json. Analizando los datos vemos que tiene Geometría tipo línea.

En este caso dejamos la mayoría de las columnas.

Vemos como reconoce bien el dataset y marca las líneas uniendo efectivamente las distintas paradas de metro. Damos estilo a la capa, cambiando el color de las líneas por el valor numerolineausuario que al igual que en el anterior dataset representan las líneas de metro. Tenemos el mismo problema que anteriormente con el número de colores, por lo que damos color a las mismas líneas de metro que en el anterior dataset dejando en otras la línea 8, 11 y ramales.

Con estas dos capas tenemos “dibujada” la red de Metro de Madrid. 
{% include figure image_path="/images/paradas.png" %}

Ahora vamos a cargar el dataset Entradas y Utilizaciones por Estaciones, que es en el que se encuentran los datos útiles para nuestro análisis. Antes de cargarlo,tenemos que incluir la geolocalización de las distintas Estaciones.

Contrastamos que se ha cargado correctamente y revisamos los datos. Tenemos información de Entradas y Utilizaciones desde 2014 hasta 2018. Para nuestro análisis vamos a centrarnos en los datos de 2018, ya que son los más actuales y pueden darnos una información más cercana a la realidad que nos ayude a decidir (Otro estudio interesante podría ser estudiar cómo ha ido aumentando la utilización de las estaciones que se han abierto en los últimos años).

Con este dataset ya podemos empezar nuestro análisis. 


Comenzamos con las Utilización de las Estaciones:
{% include figure image_path="/images/map1.jpg" %}

Podemos ver que en 2018 el total de utilizaciones está cerca de los 1.600 millones. La línea de metro más utilizada es la línea 6 con 250 millones de utilizaciones durante 2018. Le siguen la línea 1 con 210 millones y la línea 5 en tercer lugar con 160 millones.

El Máximo de Utilizaciones de una Estación está en torno a 24 millones.

Ahora vamos a ver las Entradas en las Estaciones:
{% include figure image_path="/images/map2.jpg" %}

En cuanto a las Entradas, tenemos en 2018 más de 651 millones.

El orden de las Estaciones con mayor número de Entradas no varía con respecto a las Utilizaciones. La línea 6 tiene 98 millones, 96 millones la línea 1, y 68 millones la línea 5. Podemos ver que la diferencia existente en este apartado entre la línea 1 y la 6 es considerablemente menor.

El Máximo de Entradas de una Estación, está en torno a 10 millones.

Tras analizar los datos a grandes rasgos para saber entre qué cifras nos movemos, vamos a realizar un análisis visual de ambas variables. Para ello les damos estilos.

Renombramos la capa a Entradas2018, cambiamos el tamaño de los puntos por valor según la columna entradas_2018. Cambiamos el tamaño máximo a 45, y la cuantificación a Equal Interval. Le damos color azul al punto, y aumentamos la transparencia al 75% aproximadamente.

Duplicamos la capa y la renombramos como Utilizaciones2018. Cambiamos el tamaño máximo de los puntos a 45, le otorgamos tamaño según el valor de la columna utilizaciones_2018 y la cuantificación a Equal Interval. Le damos color amarillo opaco.

Con estos estilos podemos ver de una manera gráfica las Estaciones que poseen un mayor número de Entradas y de Utilizaciones. 
{% include figure image_path="/images/map3.jpg" %}

Para decidir cuales son las estaciones que van a formar parte de nuestra selección decidimos que aquellas Estaciones que superen la mitad del Máximo de Entradas y/o Utilizaciones son Estaciones clave en el flujo de movilidad de Madrid.

Ahora con la ayuda de los Widgets vamos a filtrar por las estaciones que tengan más de 5 millones de Entradas (La mitad del Máximo) y más de 12 millones de Utilizaciones (Igualmente, la mitad del Máximo).
{% include figure image_path="/images/map4.jpg" %}

Lógicamente nos salen 5 Estaciones calve de la ciudad de Madrid, ya que algunas son Intercambiadores con Estaciones de Autobús y Renfe (Moncloa, Méndez Álvaro y Atocha Renfe), luego tenemos Sol, eje de la ciudad, y Príncipe Pío , estación de obligado paso para Madrid Sur.

Para mejorar nuestra visualización realizamos dos análisis Intercept and Aggregate entre la capa METRO MADRID y las capas Utilizaciones2018 y Entradas2018.

Gracias a estos análisis traspasamos la información de las columnas que nos interesan (utilizaciones_2018 y entradas_2018) a la capa METRO MADRID.

En la capa METRO MADRID activamos un Pop-Up flotante en oscuro, que nos aporta la información de la Estación, la suma de Entradas y de Utilizaciones en 2018.

Además de esto, hemos creado 3 Widgets:
-	Utilizaciones por Línea de Metro: Para ver las líneas con mayor número de utilizaciones y poder filtrar por ellas.
-	Entradas en las Estaciones: para poder realizar la selección final una vez hemos decidido el criterio.
-	Utilización de las Estaciones: al igual que la anterior para la selección final.

Por último, hemos creado una leyenda muy básica con la información relevante de la visualización, las Entradas y las Utilizaciones.

## 3.	Conclusiones

Tras el análisis realizado de los datos, hemos llegado a la conclusión de que las 5 estaciones con mayor transito de viajeros son, de sur a norte, Méndez Álvaro, Atocha Renfe, Sol, Príncipe Pío y Moncloa.

Dos estaciones pertenecen a la línea 6 de Metro, Méndez Álvaro y Moncloa, pero se encuentran en puntos opuestos de la misma. Tenemos en la línea 1 Atocha Renfe, en la línea 10 a Príncipe Pío, y por último a Sol que se encuentra 	en el centro de Madrid y pasan 3 líneas.

La conclusión de nuestro análisis es que Metro de Madrid podría crear una línea de tan solo 5 paradas que tuviera el recorrido Méndez Álvaro-Atocha Renfe-Sol-Príncipe Pío-Moncloa, y que fuera el Eje Central de la movilidad madrileña ya que uniría las cinco Estaciones más importantes de la ciudad.

### Esta línea podría ser la LÍNEA 0, y tendría color blanco como las estrellas de Madrid.
{% include figure image_path="/images/map5.jpg" %}


