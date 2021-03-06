---
title: "WEB SCRAPING: Creando nuestra base de datos de palas de padel"
date: 2021-03-06
tags: [WebScraping, Scraping, BaseDatos]
header:
  image: "/images/padel/palaspadel.jpg"
excerpt: "WebScraping, Scraping, BaseDatos"
classes: "wide"
mathjax: "true"

---

En este blog vamos crear nuestra base de datos de datos de palas de padel scrapeando la información de una página web.
En este caso existen 49 páginas con 12 palas en cada página, en total tendremos información de más de 580 palas.
Nos interesa tener la siguiente información por cada pala:
 - Nombre de la pala
 - Precio de la pala
 - Puntuación de la pala.
 - Composición de la pala.

Para este proceso deberemos hacer 2 pasos:
    1.- Recorreremos las 49 páginas para obtener los siguientes datos: Nombre de la pala, precio de la pala, puntuación global y link a la página de la pala.
    2.- Recorreremos los más de 580 enlaces para obtener la puntuación desglosada y la composición de cada pala.

Comenzamos el proceso


```python
# Importamos librerías necesarias
import pandas as pd
import requests
from bs4 import BeautifulSoup
import time
```

Vamos a probar en primer lugar a scrapear los datos 'Nombre', 'Puntuación' y 'Precio' de las palas de 1 única página. 


```python
# Dirección de la página web
url = "https://padelzoom.es/palas/"
# Hacemos el get requests para leer la información
page = requests.get(url)
# Analizamos la información en formato html
soup = BeautifulSoup(page.content, 'html.parser')
```

Una vez hemos definido la dirección, hacemos el get de la información y la convertimos en formato html.


Antes de comenzar a obtener la información deseada, tenemos que conocer el nombre del contenedor donde se encuentra. Para esto vamos a la página web y pinchando sobre el botón derecho seleccionamos la opción de inspeccionar.


{% include figure image_path="/images/padel/inspeccionar.png" %}

Después buscaremos el nombre del contenedor que contiene la información.

{% include figure image_path="/images/padel/contenedor.png" %}


Una vez que sabemos esto podemos empezar con el scraping.


```python
# Buscamos todas las etiquetas de la clase "text-title-price" que es el contenedor de la 
# información que estamos buscando
pl = soup.find_all('div', class_="text-title-price")

# Creamos 3 listas vacías para almacenar la información
palas = list()
puntuacion = list()
precio = list()
```


```python
# Creamos un bucle sobre toda la información que hay en cada contenedor "text-title-price"
# Y añadimos cada tipo de información en su lista.
for i in pl:
    a = i.find('p')
    b = i.find('span', class_="font-weight-600 color-red")
    c = i.find('span', class_="color-blue font-weight-600")
    palas.append(a.text)
    puntuacion.append(b.text)
    precio.append(c.text)
```


```python
#Unificamos toda la información en una lista única
lista = (palas, puntuacion, precio)
lista
```




    (['Siux Diablo Mate',
      'Bullpadel Vertex 2 2019',
      'Head Graphene 360 Alpha Pro 2019',
      'Adidas Adipower CTRL 2.0',
      'Vibora Black Mamba Black Series 1K',
      'Bullpadel Vertex 02 2020',
      'Vibora Yarara Edition Black Series 1K',
      'Vibora King Cobra Black 1K',
      'Star Vie Metheora Warrior 2020',
      'Alkemia Vento',
      'Alkemia Ignis',
      'Head Graphene 360 Alpha Pro 2021'],
     ['8.70',
      '8.70',
      '8.70',
      '8.70',
      '8.70',
      '8.70',
      '8.70',
      '8.70',
      '8.70',
      '8.70',
      '8.70',
      '8.70'],
     ['199.00',
      '158.95',
      '159.95',
      '218.95',
      '231.95',
      '187.90',
      '259.50',
      '304.50',
      '262.50',
      '149.00',
      '149.00',
      '231.90'])




```python
# Convertimos en dataframe
df = pd.DataFrame (lista).transpose()
df.columns = ['Pala','Puntuación','Precio']
df
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
      <th>Pala</th>
      <th>Puntuación</th>
      <th>Precio</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>0</td>
      <td>Siux Diablo Mate</td>
      <td>8.70</td>
      <td>199.00</td>
    </tr>
    <tr>
      <td>1</td>
      <td>Bullpadel Vertex 2 2019</td>
      <td>8.70</td>
      <td>158.95</td>
    </tr>
    <tr>
      <td>2</td>
      <td>Head Graphene 360 Alpha Pro 2019</td>
      <td>8.70</td>
      <td>159.95</td>
    </tr>
    <tr>
      <td>3</td>
      <td>Adidas Adipower CTRL 2.0</td>
      <td>8.70</td>
      <td>218.95</td>
    </tr>
    <tr>
      <td>4</td>
      <td>Vibora Black Mamba Black Series 1K</td>
      <td>8.70</td>
      <td>231.95</td>
    </tr>
    <tr>
      <td>5</td>
      <td>Bullpadel Vertex 02 2020</td>
      <td>8.70</td>
      <td>187.90</td>
    </tr>
    <tr>
      <td>6</td>
      <td>Vibora Yarara Edition Black Series 1K</td>
      <td>8.70</td>
      <td>259.50</td>
    </tr>
    <tr>
      <td>7</td>
      <td>Vibora King Cobra Black 1K</td>
      <td>8.70</td>
      <td>304.50</td>
    </tr>
    <tr>
      <td>8</td>
      <td>Star Vie Metheora Warrior 2020</td>
      <td>8.70</td>
      <td>262.50</td>
    </tr>
    <tr>
      <td>9</td>
      <td>Alkemia Vento</td>
      <td>8.70</td>
      <td>149.00</td>
    </tr>
    <tr>
      <td>10</td>
      <td>Alkemia Ignis</td>
      <td>8.70</td>
      <td>149.00</td>
    </tr>
    <tr>
      <td>11</td>
      <td>Head Graphene 360 Alpha Pro 2021</td>
      <td>8.70</td>
      <td>231.90</td>
    </tr>
  </tbody>
</table>
</div>



Vemos que hemos obtenido el resultado que queríamos, así que procedemos a scrapear sus enlaces a las páginas de cada pala donde aparecen los detalles de cada una.


```python
# Por otro lado vamos a escrapear los enlaces de las 12 palas que tenemos en esta página

# Le decimos que busque todos enlaces a otras páginas
tags = soup('a')
# Creamos la lista vacía
enlaces = list()
# Le añadimos la subcadena que vamos a utilizar como condición que tienen que tener los enlaces 
# para añadir a nuestra lista, ya que hay multiples enlaces y nosotros solo estamos interesados
# en los enlaces a las palas
sub = 'https://padelzoom.es/'

# Añadimos un contador a nuestro bucle ya que hemos detectado que nos interesan solo los 12 primeros
# enlaces de cada página, ya que pertenecen a las 12 palas que hay en cada página. Existen otros enlaces
# que también cumplirían la condición pero no nos interesan.
count = 0

# Lanzamos el bucle
for tag in tags:
    b = (tag.get('href'))
    if count < 12 and sub in b:
        enlaces.append(b)
        count += 1
    
# Comprobamos que tenemos los 12 enlaces que buscamos
enlaces
```




    ['https://padelzoom.es/siux-diablo-mate/',
     'https://padelzoom.es/bullpadel-vertex-2-2019/',
     'https://padelzoom.es/head-graphene-360-alpha-pro-2019/',
     'https://padelzoom.es/adidas-adipower-ctrl-2-0/',
     'https://padelzoom.es/vibora-black-mamba-black-series-1k/',
     'https://padelzoom.es/bullpadel-vertex-02-2020/',
     'https://padelzoom.es/vibora-yarara-edition-black-series-1k/',
     'https://padelzoom.es/vibora-king-cobra-black-1k/',
     'https://padelzoom.es/star-vie-metheora-warrior-2020/',
     'https://padelzoom.es/alkemia-vento/',
     'https://padelzoom.es/alkemia-ignis/',
     'https://padelzoom.es/head-graphene-360-alpha-pro-2021/']



Tras comprobar que efectivamente obtenemos el resultado esperado, creamos la lista de páginas en las que lanzar el scrapeo.


```python
# Ahora vamos a hacer una lista de las 49 páginas en las que tenemos que entrar para hacer el scraping
webs = list()

# Añadimos la primera página a nuestra lista de manera manual ya que tiene un formato distinto
webs.append(url)

# Observamos que el resto de páginas tienen la misma estructura, solo cambia el número de página, por 
# lo que podemos crear la lista con un bucle
for i in range(2,50):
    webs.append('https://padelzoom.es/palas/?fwp_paged='+ str(i))

# Comprobamos
webs
```




    ['https://padelzoom.es/palas/',
     'https://padelzoom.es/palas/?fwp_paged=2',
     'https://padelzoom.es/palas/?fwp_paged=3',
     'https://padelzoom.es/palas/?fwp_paged=4',
     'https://padelzoom.es/palas/?fwp_paged=5',
     'https://padelzoom.es/palas/?fwp_paged=6',
     'https://padelzoom.es/palas/?fwp_paged=7',
     'https://padelzoom.es/palas/?fwp_paged=8',
     'https://padelzoom.es/palas/?fwp_paged=9',
     'https://padelzoom.es/palas/?fwp_paged=10',
     'https://padelzoom.es/palas/?fwp_paged=11',
     'https://padelzoom.es/palas/?fwp_paged=12',
     'https://padelzoom.es/palas/?fwp_paged=13',
     'https://padelzoom.es/palas/?fwp_paged=14',
     'https://padelzoom.es/palas/?fwp_paged=15',
     'https://padelzoom.es/palas/?fwp_paged=16',
     'https://padelzoom.es/palas/?fwp_paged=17',
     'https://padelzoom.es/palas/?fwp_paged=18',
     'https://padelzoom.es/palas/?fwp_paged=19',
     'https://padelzoom.es/palas/?fwp_paged=20',
     'https://padelzoom.es/palas/?fwp_paged=21',
     'https://padelzoom.es/palas/?fwp_paged=22',
     'https://padelzoom.es/palas/?fwp_paged=23',
     'https://padelzoom.es/palas/?fwp_paged=24',
     'https://padelzoom.es/palas/?fwp_paged=25',
     'https://padelzoom.es/palas/?fwp_paged=26',
     'https://padelzoom.es/palas/?fwp_paged=27',
     'https://padelzoom.es/palas/?fwp_paged=28',
     'https://padelzoom.es/palas/?fwp_paged=29',
     'https://padelzoom.es/palas/?fwp_paged=30',
     'https://padelzoom.es/palas/?fwp_paged=31',
     'https://padelzoom.es/palas/?fwp_paged=32',
     'https://padelzoom.es/palas/?fwp_paged=33',
     'https://padelzoom.es/palas/?fwp_paged=34',
     'https://padelzoom.es/palas/?fwp_paged=35',
     'https://padelzoom.es/palas/?fwp_paged=36',
     'https://padelzoom.es/palas/?fwp_paged=37',
     'https://padelzoom.es/palas/?fwp_paged=38',
     'https://padelzoom.es/palas/?fwp_paged=39',
     'https://padelzoom.es/palas/?fwp_paged=40',
     'https://padelzoom.es/palas/?fwp_paged=41',
     'https://padelzoom.es/palas/?fwp_paged=42',
     'https://padelzoom.es/palas/?fwp_paged=43',
     'https://padelzoom.es/palas/?fwp_paged=44',
     'https://padelzoom.es/palas/?fwp_paged=45',
     'https://padelzoom.es/palas/?fwp_paged=46',
     'https://padelzoom.es/palas/?fwp_paged=47',
     'https://padelzoom.es/palas/?fwp_paged=48',
     'https://padelzoom.es/palas/?fwp_paged=49']



Efectivamente se ha creado la lista con los enlaces a las 49 páginas.

Ahora sí, vamos a lanzar un bucle que recorra las 49 páginas y nos scrapee la información de las 588 palas. Finalmente tendremos de cada pala 'Enlace', 'Modelo', 'Puntuación' y 'Precio'.


```python
# Creamos listas vacías para incluir la distinta información
palas = list()
puntuacion = list()
precio = list()
enlaces = list()

# Creamos un bucle sobre las 49 páginas
for web in webs:
    
    # Vamos a obtener toda la información que hay en cada contenedor "text-title-price"
    pl = soup.find_all('div', class_="text-title-price")
    # Hacemos el get de cada web
    page = requests.get(web)
    # Analizamos la información en formato html
    soup = BeautifulSoup(page.content, 'html.parser')
    # Creamos un bucle sobre la información que tenemos en el contenedor "text-title-price"
    # para añadir únicamente la que nos interese
    for i in pl:
        a = i.find('p')
        b = i.find('span', class_="font-weight-600 color-red")
        c = i.find('span', class_="color-blue font-weight-600")
        palas.append(a.text)
        puntuacion.append(b.text)
        precio.append(c.text)
    
    # Buscamos todos enlaces a otras páginas
    tags = soup('a')
    # Añadimos un contador a nuestro bucle ya que hemos detectado que nos interesan solo los 12 primeros
    # enlaces de cada página y no el resto aunque cumplan la condición
    count = 0
    # Añadimos la subcadena que vamos a utilizar como condición para añadir a nuestra lista
    sub = 'https://padelzoom.es/'
    for tag in tags:
        b = (tag.get('href'))
        if count < 12 and sub in b:
            enlaces.append(b)
            count += 1
            
    # Añadimos un time.sleep para no saturar la web
    time.sleep(1)
```


```python
# Unificamos toda la información en una lista única
lista = (palas, puntuacion, precio, enlaces)

# Convertimos en dataframe
df = pd.DataFrame (lista).transpose()
df.columns = ['Pala','Puntuacion','Precio','Enlaces']

# Comprobamos
df
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
      <th>Pala</th>
      <th>Puntuacion</th>
      <th>Precio</th>
      <th>Enlaces</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>0</td>
      <td>Bullpadel Hack 2019</td>
      <td>8.60</td>
      <td>179.00</td>
      <td>https://padelzoom.es/siux-diablo-mate/</td>
    </tr>
    <tr>
      <td>1</td>
      <td>Royal Padel Whip Polietileno 2019</td>
      <td>8.60</td>
      <td>132.90</td>
      <td>https://padelzoom.es/bullpadel-vertex-2-2019/</td>
    </tr>
    <tr>
      <td>2</td>
      <td>Adidas Adipower 2.0</td>
      <td>8.60</td>
      <td>198.95</td>
      <td>https://padelzoom.es/head-graphene-360-alpha-p...</td>
    </tr>
    <tr>
      <td>3</td>
      <td>Siux Diablo Granph</td>
      <td>8.60</td>
      <td>279.00</td>
      <td>https://padelzoom.es/adidas-adipower-ctrl-2-0/</td>
    </tr>
    <tr>
      <td>4</td>
      <td>Siux Diablo Grafeno Azul</td>
      <td>8.60</td>
      <td>279.00</td>
      <td>https://padelzoom.es/vibora-black-mamba-black-...</td>
    </tr>
    <tr>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <td>583</td>
      <td>Adidas V6</td>
      <td>5.90</td>
      <td>49.95</td>
      <td>https://padelzoom.es/adidas-v5/</td>
    </tr>
    <tr>
      <td>584</td>
      <td>Adidas V7</td>
      <td>5.90</td>
      <td>70.04</td>
      <td>https://padelzoom.es/primeros-pasos-padel/</td>
    </tr>
    <tr>
      <td>585</td>
      <td>Adidas Match Light 3.0 2021</td>
      <td>5.90</td>
      <td>54.00</td>
      <td>https://padelzoom.es/padel-pro-shop/</td>
    </tr>
    <tr>
      <td>586</td>
      <td>Varlion LW One Soft 2018</td>
      <td>6.10</td>
      <td></td>
      <td>https://padelzoom.es/mejores-palas-padel-junio...</td>
    </tr>
    <tr>
      <td>587</td>
      <td>Bullpadel Libra 2018</td>
      <td>6.10</td>
      <td></td>
      <td>https://padelzoom.es/star-vie-metheora-warrior...</td>
    </tr>
  </tbody>
</table>
<p>588 rows × 4 columns</p>
</div>




```python
# Guardamos la información en un excel eliminando el índice
df.to_excel('palas.xlsx',index=False)
```

# Ahora vamos con la puntuación de cada pala

En este segunda fase, vamos a obtener por una lado las características de las palas: 'Temporada', 'Material', 'Tacto',...; y por otro, vamos a obtener la puntuación de la pala: 'Salida de bola', 'Potencia', 'Manejabilidad',...

Probamos a obtener las puntuaciones de la pala Siux Diablo Mate


```python
url2 = 'https://padelzoom.es/siux-diablo-mate/'
# Hacemos el get requests para leer la información
page2 = requests.get(url2)
# Analizamos la información en formato html
soup = BeautifulSoup(page2.content, 'html.parser')
```


```python
# Buscamos todas las etiquetas de la clase "value-puntuacion" que es el contenedor de la 
# información que estamos buscando
pts = soup.find_all('div', class_="value-puntuacion")

# Creamos una lista vacía para almacenar la información
puntos = list()

# Creamos un bucle que recorra cada elemento de la lista webs
for p in pts:
    a = p.find('span')
    puntos.append(a.text)
    
# Comprobamos
puntos
```




    ['8.70', '9.00', '9.00', '9.00', '8.00', '8.50']



Vemos que hemos obtenido correctamente las puntuaciones de la pala


```python
# En este punto cargamos la tabla 'palas.xslx' que guardamos anteriormente 
palas = pd.read_excel('palas.xlsx')
```


```python
# Rescatamos la columna enlaces para hacer un bucle
enlaces_palas = palas['Enlaces']
enlaces_palas
```




    0              https://padelzoom.es/bullpadel-hack-2019/
    1      https://padelzoom.es/royal-padel-whip-polietil...
    2              https://padelzoom.es/adidas-adipower-2-0/
    3               https://padelzoom.es/siux-diablo-granph/
    4         https://padelzoom.es/siux-diablo-grafeno-azul/
                                 ...                        
    583                      https://padelzoom.es/adidas-v6/
    584                      https://padelzoom.es/adidas-v7/
    585    https://padelzoom.es/adidas-match-light-3-0-2021/
    586       https://padelzoom.es/varlion-lw-one-soft-2018/
    587           https://padelzoom.es/bullpadel-libra-2018/
    Name: Enlaces, Length: 588, dtype: object



Vamos a lanzar el bucle para scrapear las puntuaciones


```python
# Creamos una lista vacía para almacenar la información
puntos = list()
    
# Y lanzamos el bucle para obtener las puntuaciones
for enlace in enlaces_palas:
        
    url = enlace
    page = requests.get(url)
    # Analizamos la información en formato html
    soup = BeautifulSoup(page.content, 'html.parser')
    
    # Creamos uns lista vacía para almancenar la información de cada página, y que se vacíe 
    # en cada visita a una nueva página
    puntos_pala = list()
    
    # Añadimos el nombre de cada pala en primer lugar
    modelo = soup.find('span', property="name")
    puntos_pala.append(modelo.text)   
        
    # Y ahora las puntuaciones
    pts = soup.find_all('div', class_="value-puntuacion")
    # Creamos un bucle que recorra cada elemento
    for pt in pts:
        p = pt.find('span')
        puntos_pala.append(p.text)
    puntos.append(puntos_pala)

    time.sleep(0.3)
```


```python
# Creamos un dataframe con toda la información
data = pd.DataFrame(puntos, columns = ('Modelo','Total','Potencia','Control','Salida de bola','Manejabilidad','Punto dulce'))

# Comprobamos
data
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
      <th>Modelo</th>
      <th>Total</th>
      <th>Potencia</th>
      <th>Control</th>
      <th>Salida de bola</th>
      <th>Manejabilidad</th>
      <th>Punto dulce</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>0</td>
      <td>Bullpadel Hack 2019</td>
      <td>8.60</td>
      <td>9.00</td>
      <td>8.00</td>
      <td>9.50</td>
      <td>8.50</td>
      <td>8.00</td>
    </tr>
    <tr>
      <td>1</td>
      <td>Royal Padel Whip Polietileno 2019</td>
      <td>8.60</td>
      <td>7.00</td>
      <td>9.50</td>
      <td>8.50</td>
      <td>9.00</td>
      <td>9.00</td>
    </tr>
    <tr>
      <td>2</td>
      <td>Adidas Adipower 2.0</td>
      <td>8.60</td>
      <td>9.50</td>
      <td>9.00</td>
      <td>8.00</td>
      <td>8.50</td>
      <td>8.00</td>
    </tr>
    <tr>
      <td>3</td>
      <td>Siux Diablo Granph</td>
      <td>8.60</td>
      <td>9.00</td>
      <td>8.50</td>
      <td>8.50</td>
      <td>8.00</td>
      <td>9.00</td>
    </tr>
    <tr>
      <td>4</td>
      <td>Siux Diablo Grafeno Azul</td>
      <td>8.60</td>
      <td>9.50</td>
      <td>8.00</td>
      <td>8.50</td>
      <td>8.50</td>
      <td>8.50</td>
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
    </tr>
    <tr>
      <td>583</td>
      <td>Adidas V6</td>
      <td>5.90</td>
      <td>5.00</td>
      <td>5.50</td>
      <td>6.50</td>
      <td>7.00</td>
      <td>5.50</td>
    </tr>
    <tr>
      <td>584</td>
      <td>Adidas V7</td>
      <td>5.90</td>
      <td>5.00</td>
      <td>5.50</td>
      <td>6.50</td>
      <td>7.00</td>
      <td>5.50</td>
    </tr>
    <tr>
      <td>585</td>
      <td>Adidas Match Light 3.0 2021</td>
      <td>5.90</td>
      <td>5.00</td>
      <td>6.50</td>
      <td>5.50</td>
      <td>7.50</td>
      <td>5.00</td>
    </tr>
    <tr>
      <td>586</td>
      <td>Varlion LW One Soft 2018</td>
      <td>6.10</td>
      <td>5.00</td>
      <td>6.00</td>
      <td>6.00</td>
      <td>7.50</td>
      <td>6.00</td>
    </tr>
    <tr>
      <td>587</td>
      <td>Bullpadel Libra 2018</td>
      <td>6.10</td>
      <td>5.00</td>
      <td>5.50</td>
      <td>6.00</td>
      <td>8.00</td>
      <td>6.00</td>
    </tr>
  </tbody>
</table>
<p>588 rows × 7 columns</p>
</div>




```python
# Guardamos el dataframe en un excel
data.to_excel('puntos.xlsx',index=False)
```

Tras haber obtenido las puntuaciones, vamos a scrapear las características de las palas

Primero vamos a hacer algunas comprobaciones con una única pala como antes


```python
url3 = 'https://padelzoom.es/siux-diablo-mate/'
# Hacemos el get requests para leer la información
page3 = requests.get(url3)
# Analizamos la información en formato html
soup = BeautifulSoup(page3.content, 'html.parser')
```


```python
# Buscamos todas las etiquetas de la clase "text-title-price" que es el contenedor de la 
# información que estamos buscando
caract = soup.find_all('div', class_="col-md-4 col-sm-6")

# Creamos listas vacías para almacenar la información
names = list()
caracteristicas = list()

# Creamos un bucle que recorra cada elemento de la lista webs
for cr in caract:
    a = cr.find('span', property="name")
    b = cr.find('div', class_="description-pala")
    c = b.find_all('p')
    for i in c:
        caracteristicas.append(i.text)
    
        
        
caracteristicas

```




    ['Temporada : 2020',
     'Material marco : Fibra de carbono 3K',
     'Material plano : Fibra de carbono y aluminio',
     'Material goma : EVA High Memory',
     'Tacto : Medio-Duro',
     'Forma : Diamante',
     'Peso : 360-375 gramos']




```python
# Creamos una función para eliminar la información que no necesitamos
def eliminar_puntos(lista):
    return [l.split(' : ')[1] for l in lista]

eliminar_puntos(caracteristicas)
```




    ['2020',
     'Fibra de carbono 3K',
     'Fibra de carbono y aluminio',
     'EVA High Memory',
     'Medio-Duro',
     'Diamante',
     '360-375 gramos']




```python
# Creamos una lista vacía para almacenar la información
caracteristicas = list()
    
# Y lanzamos el bucle para obtener las caracteristicas
for enlace in enlaces_palas:
        
    url = enlace
    page = requests.get(url)
    # Analizamos la información en formato html
    soup = BeautifulSoup(page.content, 'html.parser')
    
    # Creamos uns lista vacía para almancenar la información de cada página, y que se vacie 
    # en cada visita a una web distinta
    caract_pala = list()
        
    # Buscamos todas las etiquetas de la clase "col-md-4 col-sm-6" que es el contenedor de la 
    # información que estamos buscando
    caract = soup.find_all('div', class_="col-md-4 col-sm-6")
   
    # Creamos un bucle que recorra cada elemento de caract   
    for cr in caract:
        # Añadimos el nombre de cada pala en primer lugar
        a = cr.find('span', property="name")
        caract_pala.append(a.text)
        
        # Y ahora las puntuaciones
        b = cr.find('div', class_="description-pala")
        # Creamos un bucle que recorra cada elemento
        c = b.find_all('p')
        for i in c:
            caract_pala.append(i.text.split(' : ')[1])
    # Añadimos a nuestra lista
    caracteristicas.append(caract_pala)

    time.sleep(0.2)

caracteristicas
```




    [['Bullpadel Hack 2019',
      '2019',
      'Tubular 100% carbono + protector de aluminio',
      'Xtend Carbon 18K',
      'Black EVA',
      'Medio-Blando',
      'Diamante',
      '365-380 gramos'],
     ['Royal Padel Whip Polietileno 2019',
      '2019',
      'Tubular bidireccional de fibra de vidrio con refuerzos de tejido de carbono.',
      'Tejido de vidrio aluminizado. Impregnación de resina epoxi incluyendo dióxido de titanio',
      'Polietileno de alta densidad',
      'Medio-Blando',
      'Redonda',
      '360-385 gramos'],
     ['Adidas Adipower 2.0',
      '2020',
      'Fibra de carbono 3K',
      'Fibra de carbono y aluminio',
      'EVA High Memory',
      'Medio-Duro',
      'Diamante',
      '360-375 gramos'],
     ['Siux Diablo Granph',
      '2019',
      'Carbono + Grafeno',
      'Carbono + Grafeno',
      'EVA Soft',
      'Medio-Duro',
      'Lágrima',
      '360-375 gramos'],
     ...
     ...
     ['Adidas V7',
      '2020',
      'Fibra de carbono',
      'Fibra de vidrio',
      'EVA Soft',
      'Medio-Blando',
      'Lágrima',
      '360-375 gramos'],
     ['Adidas Match Light 3.0 2021',
      '2021',
      'Fibra de vidrio',
      'Fibra de vidrio',
      'EVA Soft',
      'Medio-Blando',
      'Diamante',
      '350-360 gramos'],
     ['Varlion LW One Soft 2018',
      '2018',
      'Fibra de vidrio',
      'Fibra de vidrio',
      'SYL Core',
      'Medio',
      'Redonda',
      '325-355 gramos'],
     ['Bullpadel Libra 2018',
      '2018',
      'Tubual 100% Carbono',
      'Polyglass',
      'Evalastic',
      'Medio-Blando',
      'Redonda',
      '350-360 gramos']]




```python
# Creamos un dataframe con toda la información
data_caract = pd.DataFrame(caracteristicas, columns = ('Modelo','Temporada','Material Marco','Material plano',
                                                       'Material goma','Tacto','Forma','Peso'))

# Comprobamos
data_caract
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
      <th>Modelo</th>
      <th>Temporada</th>
      <th>Material Marco</th>
      <th>Material plano</th>
      <th>Material goma</th>
      <th>Tacto</th>
      <th>Forma</th>
      <th>Peso</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>0</td>
      <td>Bullpadel Hack 2019</td>
      <td>2019</td>
      <td>Tubular 100% carbono + protector de aluminio</td>
      <td>Xtend Carbon 18K</td>
      <td>Black EVA</td>
      <td>Medio-Blando</td>
      <td>Diamante</td>
      <td>365-380 gramos</td>
    </tr>
    <tr>
      <td>1</td>
      <td>Royal Padel Whip Polietileno 2019</td>
      <td>2019</td>
      <td>Tubular bidireccional de fibra de vidrio con r...</td>
      <td>Tejido de vidrio aluminizado. Impregnación de ...</td>
      <td>Polietileno de alta densidad</td>
      <td>Medio-Blando</td>
      <td>Redonda</td>
      <td>360-385 gramos</td>
    </tr>
    <tr>
      <td>2</td>
      <td>Adidas Adipower 2.0</td>
      <td>2020</td>
      <td>Fibra de carbono 3K</td>
      <td>Fibra de carbono y aluminio</td>
      <td>EVA High Memory</td>
      <td>Medio-Duro</td>
      <td>Diamante</td>
      <td>360-375 gramos</td>
    </tr>
    <tr>
      <td>3</td>
      <td>Siux Diablo Granph</td>
      <td>2019</td>
      <td>Carbono + Grafeno</td>
      <td>Carbono + Grafeno</td>
      <td>EVA Soft</td>
      <td>Medio-Duro</td>
      <td>Lágrima</td>
      <td>360-375 gramos</td>
    </tr>
    <tr>
      <td>4</td>
      <td>Siux Diablo Grafeno Azul</td>
      <td>2019</td>
      <td>Carbono + Grafeno</td>
      <td>Carbono</td>
      <td>Black EVA Soft</td>
      <td>Medio-Duro</td>
      <td>Lágrima</td>
      <td>360-375 gramos</td>
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
    </tr>
    <tr>
      <td>583</td>
      <td>Adidas V6</td>
      <td>2019</td>
      <td>Fibra de carbono</td>
      <td>Fibra de vidrio</td>
      <td>EVA Soft</td>
      <td>Medio-Blando</td>
      <td>Lágrima</td>
      <td>360-375 gramos</td>
    </tr>
    <tr>
      <td>584</td>
      <td>Adidas V7</td>
      <td>2020</td>
      <td>Fibra de carbono</td>
      <td>Fibra de vidrio</td>
      <td>EVA Soft</td>
      <td>Medio-Blando</td>
      <td>Lágrima</td>
      <td>360-375 gramos</td>
    </tr>
    <tr>
      <td>585</td>
      <td>Adidas Match Light 3.0 2021</td>
      <td>2021</td>
      <td>Fibra de vidrio</td>
      <td>Fibra de vidrio</td>
      <td>EVA Soft</td>
      <td>Medio-Blando</td>
      <td>Diamante</td>
      <td>350-360 gramos</td>
    </tr>
    <tr>
      <td>586</td>
      <td>Varlion LW One Soft 2018</td>
      <td>2018</td>
      <td>Fibra de vidrio</td>
      <td>Fibra de vidrio</td>
      <td>SYL Core</td>
      <td>Medio</td>
      <td>Redonda</td>
      <td>325-355 gramos</td>
    </tr>
    <tr>
      <td>587</td>
      <td>Bullpadel Libra 2018</td>
      <td>2018</td>
      <td>Tubual 100% Carbono</td>
      <td>Polyglass</td>
      <td>Evalastic</td>
      <td>Medio-Blando</td>
      <td>Redonda</td>
      <td>350-360 gramos</td>
    </tr>
  </tbody>
</table>
<p>588 rows × 8 columns</p>
</div>




```python
# Guardamos el dataframe en un excel
data_caract.to_excel('caracteristicas.xlsx',index=False)
```

Con esto ya tenemos nuestra base de datos de palas. Ahora ya podemos hacer análisis o algún algoritmo, pero eso lo dejamos para un siguiente artículo.
