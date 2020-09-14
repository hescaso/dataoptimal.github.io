# Análisis de sentimientos con reviews de productos de Amazon España (opcional)

El dataset que utilizarás en este ejercicio (que no es obligatorio entregar) lo he generado utilizando `Scrapy` y `BeautifulSoup` (dos bibliotecas de Python para hacer [Web Scraping](https://es.wikipedia.org/wiki/Web_scraping)), y contiene unas $700.000$ entradas de reviews de productos de amazon.es; contiene dos columnas: el número de estrellas dadas por un usuario a un determinado producto y el comentario sobre dicho producto; exactamente igual que en el ejercico de scraping.

Ahora, tu objetivo es utilizar técnicas de procesamiento de lenguaje natural para hacer un clasificador que sea capaz de distinguir (¡y predecir!) si un comentario es positivo o negativo.

Es un ejercicio MUY complicado, más que nada porque versa sobre técnicas que no hemos visto en clase. Así que si quieres resolverlo, te va a tocar estudiar y *buscar por tu cuenta*; exactamente igual que como sería en un puesto de trabajo. Dicho esto, daré un par de pistas:

+ El número de estrellas que un usuario da a un producto es el indicador de si a dicho usuario le ha gustado el producto o no. Una persona que da 5 estrellas (el máximo) a un producto probablemente esté contenta con él, y el comentario será por tanto positivo; mientras que cuando una persona da 1 estrella a un producto es porque no está satisfecha... 
+ Teniendo el número de estrellas podríamos resolver el problema como si fuera de regresión; pero vamos a establecer una regla para convertirlo en problema de clasificación: *si una review tiene 4 o más estrellas, se trata de una review positiva; y será negativa si tiene menos de 4 estrellas*. Así que probablemente te toque transformar el número de estrellas en otra variable que sea *comentario positivo/negativo*.

Y... poco más. Lo más complicado será convertir el texto de cada review en algo que un clasificador pueda utilizar y entender (puesto que los modelos no entienden de palabras, sino de números). Aquí es donde te toca investigar las técnicas para hacerlo. El ejercicio se puede conseguir hacer, y obtener buenos resultados, utilizando únicamente Numpy, pandas y Scikit-Learn; pero siéntete libre de utilizar las bibliotecas que quieras.

Ahora escribiré una serie de *keywords* que probablemente te ayuden a saber qué buscar:

`bag of words, tokenizer, tf, idf, tf-idf, sklearn.feature_extraction, scipy.sparse, NLTK (opcional), stemmer, stop-words removal, bigrams, trigrams`

No te desesperes si te encuentras muy perdido/a y no consigues sacar nada. Tras la fecha de entrega os daré un ejemplo de solución explicado con todo el detalle posible.

¡Ánimo y buena suerte!


```python
# Importamos librerías necesarias para el análisis
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from nltk.corpus import stopwords
from nltk import word_tokenize
from nltk.stem import SnowballStemmer
from string import punctuation
from sklearn.model_selection import train_test_split
```


```python
# Leemos el archivo:
df = pd.read_csv("C:\\Users\\hesca\\Documents\\MASTER DATAHACK\\Machine Learning\\Practicas\\amazon_es_reviews.csv",
                   sep = ";", encoding ='utf-8')
```


```python
# Exploramos el dataset para ver que se ha cargado adecuadamente:
df.head(10)
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
      <th>comentario</th>
      <th>estrellas</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>0</td>
      <td>Para chicas es perfecto, ya que la esfera no e...</td>
      <td>4.0</td>
    </tr>
    <tr>
      <td>1</td>
      <td>Muy floja la cuerda y el anclaje es de mala ca...</td>
      <td>1.0</td>
    </tr>
    <tr>
      <td>2</td>
      <td>Razonablemente bien escrito, bien ambientado, ...</td>
      <td>3.0</td>
    </tr>
    <tr>
      <td>3</td>
      <td>Hola! No suel o escribir muchas opiniones sobr...</td>
      <td>5.0</td>
    </tr>
    <tr>
      <td>4</td>
      <td>A simple vista m parecia una buena camara pero...</td>
      <td>1.0</td>
    </tr>
    <tr>
      <td>5</td>
      <td>NI para pasar el rato, los personajes no tiene...</td>
      <td>1.0</td>
    </tr>
    <tr>
      <td>6</td>
      <td>el fabricante decia que es compatible con la d...</td>
      <td>2.0</td>
    </tr>
    <tr>
      <td>7</td>
      <td>el libro está en muy buenas condiciones, pero ...</td>
      <td>3.0</td>
    </tr>
    <tr>
      <td>8</td>
      <td>buen aspecto, pero le falta fortaleza. util pa...</td>
      <td>3.0</td>
    </tr>
    <tr>
      <td>9</td>
      <td>Explica de forma simple y sencilla los pensami...</td>
      <td>5.0</td>
    </tr>
  </tbody>
</table>
</div>




```python
df.describe()
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
      <th>estrellas</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>count</td>
      <td>702446.000000</td>
    </tr>
    <tr>
      <td>mean</td>
      <td>3.372171</td>
    </tr>
    <tr>
      <td>std</td>
      <td>1.435783</td>
    </tr>
    <tr>
      <td>min</td>
      <td>1.000000</td>
    </tr>
    <tr>
      <td>25%</td>
      <td>2.000000</td>
    </tr>
    <tr>
      <td>50%</td>
      <td>4.000000</td>
    </tr>
    <tr>
      <td>75%</td>
      <td>5.000000</td>
    </tr>
    <tr>
      <td>max</td>
      <td>5.000000</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Ahora vamos transformar la feature estrellas en otra binaria "postivo_negativo", donde los comentarios con :
df_positivo_negativo = []
for i in df["estrellas"]:
    if i > 3:
        df_positivo_negativo.append(1) 
    else:
        df_positivo_negativo.append(0) 
        
df["positivo_negativo"] = df_positivo_negativo
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
      <th>comentario</th>
      <th>estrellas</th>
      <th>positivo_negativo</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>0</td>
      <td>Para chicas es perfecto, ya que la esfera no e...</td>
      <td>4.0</td>
      <td>1</td>
    </tr>
    <tr>
      <td>1</td>
      <td>Muy floja la cuerda y el anclaje es de mala ca...</td>
      <td>1.0</td>
      <td>0</td>
    </tr>
    <tr>
      <td>2</td>
      <td>Razonablemente bien escrito, bien ambientado, ...</td>
      <td>3.0</td>
      <td>0</td>
    </tr>
    <tr>
      <td>3</td>
      <td>Hola! No suel o escribir muchas opiniones sobr...</td>
      <td>5.0</td>
      <td>1</td>
    </tr>
    <tr>
      <td>4</td>
      <td>A simple vista m parecia una buena camara pero...</td>
      <td>1.0</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>



Revisando por internet entiendo que no es la mejor opción lanzar un for en un dataframe, aunque en este caso sale bien.
Deberíamos haber utilizado una función apply:

**df["positivos_negativo"] = df["estrellas"].apply(lambda x:1 if (x>3) else 0)**


```python
# Vemos cuantos comentarios positivos y negativos hay para poder hacer la selección óptima de la métrica:
print(df["positivo_negativo"].value_counts())
```

    1    363735
    0    338711
    Name: positivo_negativo, dtype: int64
    

Hay un 51,78% de comentarios positivos y un 48,22% de comentarios negativos, por lo tanto podemos utilizar la métrica **Accuracy**.


```python
# Convertimos en array nuestras columnas del dataframe para agilizar los cálculos:
comentarios = df['comentario'].values
notas = df['positivo_negativo'].values
```


```python
# Separamos en train (75%) y test (25%)
comentarios_train, comentarios_test, notas_train, notas_test = train_test_split(
   comentarios, notas, test_size=0.25, random_state=8)
```

## Comenzamos con el proceso de Vectorización de los comentarios:


```python
# Creamos una bolsa de palabras y definimos la puntuación que eliminaremos de los comentarios:

spanish_stopwords = stopwords.words('spanish')
stemmer = SnowballStemmer('spanish')
non_words = list(punctuation)
non_words.extend(['¿', '¡', ',','...','``'])
non_words.extend(map(str,range(10)))

```

Definimos las funciones dentro del vectorizer para crear los tokens, limpiar los tokens y hacer el steem:


```python
def stem_tokens (tokens, stemmer):
    stemmed = []
    for item in tokens:
        stemmed.append(stemmer.stem(item))
    return stemmed

def clean_data (tokens, stop_words = ()):
    clean_tokens = []
    for token in tokens:
        if token.lower() not in spanish_stopwords and token not in non_words:
            clean_tokens.append (token)
    return clean_tokens

def tokear(text):
    tokens = []
    text = ''.join([c for c in text if c not in non_words]) # Limpieza del texto eliminando 
    tokens =  word_tokenize(text)
    tokens_limpios = clean_data(tokens)
    tokens_stemmed = stem_tokens(tokens_limpios, stemmer)
    return tokens_stemmed
```


```python
# Definimos el vectorizer con los siguientes pasos:
# - Tokenizamos para convertir cada cadena de texto en un token.
# - Convertimos las palabras en minúsculas.
# - Removemos palabras que son muy frecuentes en castellano, pero que no aportan valor semántico.
# - Hacemos el Steem, es decir, convertimos cada palabra en su raíz. (Podríamos realizar la Lematización, para convertir
# cada palabra en su lema, es decir en la palabra tal y como la encontratíamos en el diccionario, pero en este caso 
# hemos realizado el steem).

vectorizer = CountVectorizer(
                analyzer = 'word',
                tokenizer = tokear,
                lowercase = True,
                stop_words = spanish_stopwords)
```


```python
# Antes de continuar vamor a realizar una prueba de nuestras funciones con una muestra de 100 comentarios:
tokens = tokear(df["comentario"][:100])
```


```python
# Observamos si se realiza correctamente la tokenización:
tokens
```




    ['chic',
     'perfect',
     'esfer',
     'grand',
     'corre',
     'adapt',
     'muñec',
     'fin',
     'pelin',
     'gord',
     'gust',
     'carg',
     'movimient',
     'dur',
     'despues',
     '1-2',
     'dias',
     'llev',
     'para.muy',
     'floj',
     'cuerd',
     'anclaj',
     'mal',
     'calid',
     'metal',
     'dobl',
     'facil',
     'recomiendorazon',
     'bien',
     'escrit',
     'bien',
     'ambient',
     'quizas',
     'previs',
     'histori',
     'interes',
     'personaj',
     'bien',
     'dibuj',
     'parec',
     'libr',
     'recomendable.hol',
     'suel',
     'escrib',
     'much',
     'opinion',
     'product',
     'compr',
     'verd',
     'product',
     'merec',
     'desencant',
     'maquin',
     'afeit',
     'hac',
     'años',
     'compr',
     'philips',
     'cuchill',
     'rotatori',
     'piel',
     'extrem',
     'sensibl',
     'usar',
     'afeit',
     'hic',
     'polv',
     'car',
     'vam',
     'carn',
     'viv',
     'dej',
     'volv',
     'usar',
     'vez',
     'mism',
     'part',
     'apur',
     'bien',
     'barb',
     'asi',
     'despues',
     'mal',
     'experient',
     'pas',
     'cuchill',
     'gillett',
     'mach3',
     'verd',
     'iban',
     'mejor',
     'dej',
     'car',
     'ardor',
     'dur',
     'dia',
     'hac',
     'vari',
     'seman',
     'leyend',
     'for',
     'demas',
     'vi',
     'afeit',
     'cuchill',
     'lamin',
     'funcion',
     'mejor',
     'piel',
     'delic',
     'ademas',
     'apur',
     'asi',
     'dediqu',
     'investig',
     'decid',
     'compr',
     'dud',
     'si',
     'compr',
     'braun',
     'panasonic',
     'obvi',
     'decid',
     'segund',
     'consegu',
     'prestacion',
     'potenci',
     'panasonic',
     'braun',
     'pag',
     'dobl',
     'sab',
     'si',
     'final',
     'iba',
     'ir',
     'bien',
     'decid',
     'panasonic',
     'experient',
     'despues',
     'vari',
     'seman',
     'uso',
     'excelent',
     'maquin',
     'buen',
     'calid',
     'material',
     'robust',
     'ergonom',
     'result',
     'apur',
     'buenisim',
     'potent',
     'encim',
     'bat',
     'carg',
     'tan',
     'sol',
     'hor',
     'llev',
     'cinc',
     'afeit',
     'sig',
     'bat',
     'maravill',
     'limpi',
     'sup',
     'facil',
     'pued',
     'usar',
     'sec',
     'moj',
     'debaj',
     'duch',
     'espum',
     'gel',
     'afeit',
     'ah',
     'import',
     'dañ',
     'absolut',
     'piel',
     'rojec',
     'escozor',
     'tra',
     'recort',
     'patill',
     'bigot',
     'aun',
     'usad',
     'afeit',
     'bien',
     'cuell',
     'car',
     'dud',
     'recom',
     'cien',
     'cien',
     'pas',
     'ver',
     'mal',
     'apur',
     'cuchill',
     'rotatori',
     'philips.',
     'simpl',
     'vist',
     'm',
     'pareci',
     'buen',
     'cam',
     'm',
     'decepcion',
     'mux',
     'prim',
     'bañ',
     'entro',
     'agu',
     'adi',
     'cam',
     'k',
     'prob',
     'primer',
     'acuari',
     'd',
     'cas',
     '.m',
     'yev',
     'gran',
     'decepcion',
     '..ni',
     'pas',
     'rat',
     'personaj',
     'ningun',
     'dialog',
     'sos',
     'protagon',
     'falt',
     'cinc',
     'hervor',
     'narrat',
     'exist',
     'fiabil',
     'histor',
     'total',
     'nula.el',
     'fabric',
     'deci',
     'compat',
     'd610',
     'maner',
     'igual',
     'hag',
     'mal',
     'compact',
     'si',
     'funciona.el',
     'libr',
     'buen',
     'condicion',
     'inclu',
     'cd',
     'audi',
     'sol',
     'serv',
     'mediasbu',
     'aspect',
     'falt',
     'fortalez',
     'util',
     'marc',
     'cort',
     'circul',
     'veo',
     'poc',
     'fuerz',
     'resistent',
     'hac',
     'mas',
     'pruebas.expl',
     'form',
     'simpl',
     'sencill',
     'pensamient',
     'grand',
     'filosof',
     'grand',
     'corrient',
     'permit',
     'entend',
     'concept',
     'evolucion',
     'pas',
     'pas',
     'histori',
     'simpl',
     'entretenidarel',
     'calid',
     'preci',
     'buen',
     'son',
     'fuert',
     'man',
     'libr',
     'escuch',
     'perfect',
     'lector',
     'usb',
     'funcion',
     'bien',
     'conexion',
     'radi',
     'movil',
     'sencill',
     'conect',
     'radi',
     'tambienel',
     'funcion',
     'product',
     'correct',
     'buen',
     'diseñ',
     'tamañ',
     'correct',
     'calid',
     'preci',
     'acord',
     'esperadoest',
     'bien',
     'aunqu',
     'diferent',
     'presentd',
     'poquit',
     'caroy',
     'si',
     '3d',
     'genial',
     'niñ',
     'concienci',
     'mux',
     'mayor',
     'peli',
     'buen',
     'efect',
     'ser',
     'dibuj',
     'buen',
     'calidadconsider',
     'histori',
     'entreten',
     'llen',
     'creativ',
     'cont',
     'lenguaj',
     'clar',
     'armoni',
     'flu',
     'denot',
     'autor',
     'pose',
     'vast',
     'conoc',
     'cultur',
     'puebl',
     'nordic',
     'empec',
     'leerl',
     'cautiv',
     'conoc',
     'destin',
     'ivarr',
     'si',
     'gust',
     'desarroll',
     'descenlace.cu',
     'van',
     'mand',
     'sirv',
     'prevision',
     'recib',
     'inform',
     'plaz',
     'entreg',
     'cas',
     'fij',
     'fech',
     'previst',
     'deb',
     'cumpl',
     'si',
     'culp',
     'editorial',
     'cre',
     'buen',
     'polit',
     'car',
     'client',
     'vuelv',
     'hac',
     'caso.recomend',
     'si',
     'amant',
     'clasic',
     'edicion',
     'bien',
     'contien',
     'peli',
     'par',
     'idiom',
     'tres',
     'mas',
     'apr',
     'conten',
     'extras',
     'maloq',
     'ue',
     'edicion',
     'bden',
     'estet',
     'comod',
     'ajust',
     'esper',
     'calz',
     'result',
     'comod',
     'cuant',
     'usas.est',
     'marc',
     'demasi',
     'renombr',
     'calid',
     'diseñ',
     'tipic',
     'sobri',
     'encuadern',
     'bien',
     'hoj',
     'demasi',
     'gramaj',
     'pued',
     'transparent',
     'si',
     'escrib',
     'tint',
     'cuid',
     'tamañ',
     'aunqu',
     'pong',
     'tamañ',
     'grand',
     'realid',
     'a5un',
     'muj',
     'millonari',
     'elizabeth',
     'taylor',
     've',
     'ventan',
     'asesinat',
     'viej',
     'caseron',
     'enfrent',
     'mar',
     'laurenc',
     'harvey',
     'mejor',
     'amig',
     'cre',
     'asim',
     'polic',
     'recurr',
     'insistent',
     'sig',
     'insist',
     'exasper',
     'polic',
     'final',
     'cre',
     'muj',
     'alucin',
     'cabal',
     'aunqu',
     'pelicul',
     'dej',
     'monton',
     'cab',
     'suelt',
     'pued',
     'neg',
     'entretien',
     'maxim',
     'ultim',
     '20',
     'minut',
     'da',
     'gir',
     'bastant',
     'inesper',
     'ningun',
     'obra',
     'maestr',
     'men',
     'si',
     'quier',
     'pas',
     'rat',
     'pelicul',
     'conseguira.',
     'pen',
     'piez',
     'tabler',
     'carton',
     'dobl',
     'esquin',
     'facil',
     'siend',
     'jueg',
     'viaje.diseñ',
     'bonit',
     'buen',
     'preci',
     'montaj',
     'demasi',
     'complej',
     'opinion',
     'part',
     'podr',
     'mejor',
     'tornill',
     'atornill',
     'direct',
     'chasis',
     'plastic',
     'tamañ',
     'bastant',
     'pequeñ',
     'just',
     '2/3',
     'años.cumpl',
     'funcion',
     'perfect',
     'preci',
     'estupend',
     'uso',
     'micr',
     'pc',
     'habl',
     'skype',
     'similar',
     'oye',
     'clar',
     'ruid',
     'si',
     'requier',
     'equip',
     'profesional',
     'recom',
     'uso',
     'oficin',
     'cas',
     'prob',
     'exterior',
     'asi',
     'sensibl',
     'ruid',
     'ambientales.l',
     'sill',
     'buen',
     'calid',
     'buen',
     'acab',
     'embarg',
     'devolv',
     'encontr',
     'defect',
     'imped',
     'normal',
     'funcion',
     'problem',
     'amazon',
     'devolverlo.tod',
     'satisfactori',
     'plaz',
     'entreg',
     'estipul',
     '.muy',
     'recomend',
     'cumpl',
     'tod',
     'caracterist',
     'anunci',
     'contentoel',
     'product',
     'fot',
     'facil',
     'manej',
     'buen',
     'calid',
     'recomend',
     '100',
     'dud',
     'volv',
     'comprar.n',
     'ningun',
     'peg',
     'destac',
     'aunqu',
     'vez',
     'puest',
     'algun',
     'zon',
     'qued',
     'poquit',
     'torc',
     'huec',
     'cam',
     'conexion',
     'cabl',
     'cargador',
     'coincid',
     'perfect',
     'boton',
     'bloqu',
     'pantall',
     'flechit',
     'sub',
     'baj',
     'volum',
     'cubr',
     'fund',
     'ahor',
     'mas',
     'dur',
     'puls',
     'esper',
     'fund',
     'mas',
     'blandit',
     'elast',
     'amortigu',
     'caid',
     'bastant',
     'rig',
     'men',
     'mas',
     'proteg',
     'qued',
     'movil',
     'fund',
     'da',
     'sensacion',
     'si',
     'cae',
     'raj',
     'lad',
     'hech',
     'call',
     'much',
     'altur',
     'movil',
     'fund',
     'raj',
     'pantall',
     'aunqu',
     'sig',
     'siend',
     'funcional.hac',
     'comet',
     'principal',
     'mol',
     'lumbrer',
     'diseñ',
     'deb',
     'hab',
     'pens',
     'tap',
     'sac',
     'derram',
     'caf',
     'interior',
     'pues',
     'llev',
     'lengüet',
     "''",
     'entra',
     'dentr',
     'molinill',
     'engorr',
     'ahi',
     'darl',
     'sol',
     'estrell',
     'envi',
     'embalaj',
     'estrellas.com',
     'dvd',
     'tempor',
     'region',
     'ped',
     'product',
     'pes',
     'avis',
     'relat',
     'region',
     'aparec',
     'algun',
     'opinion',
     'usuari',
     'lleg',
     'disc',
     'indic',
     'region',
     'asi',
     'abrirl',
     'escrib',
     'proveedor',
     'traves',
     'amazon',
     'indic',
     'prob',
     'insist',
     'podr',
     'devolv',
     'si',
     'funcion',
     'dec',
     'funcion',
     'atencion',
     'marveli',
     'sid',
     'impec',
     'unic',
     'fall',
     "''",
     'doblaj',
     'español',
     'sudamerican',
     'suen',
     'rar',
     'aqu',
     'demas',
     'perfecto.està',
     'bien',
     'cumpl',
     'funcion',
     'falt',
     'cierr',
     'seri',
     'perfect',
     'b',
     'b',
     'b',
     'b',
     'b',
     'b',
     'b',
     'bproduct',
     '100',
     'recomend',
     'ideal',
     'orden',
     'toshib',
     'seri',
     'c55',
     'perfect',
     'bolsill',
     'lateral',
     'vien',
     'genial',
     'guard',
     'cargador',
     'raton.el',
     'dia',
     'septiembr',
     'realiz',
     'ped',
     'pag',
     '10€',
     'gast',
     'envi',
     'lug',
     'gast',
     'envi',
     'estand',
     'envi',
     'dia',
     'siguient',
     'paquet',
     'lleg',
     'dias',
     'gust',
     'devolv',
     'gast',
     'envi',
     'equipar',
     'gast',
     'envi',
     'estand',
     'product',
     'lleg',
     'dia',
     'lleg',
     'juev',
     'septiembr',
     'graci',
     'demas',
     'product',
     'adecu',
     'necesidades.muy',
     'recomend',
     'viaj',
     'quier',
     'carg',
     'pes',
     'tripod',
     'sup',
     'resistent',
     'aguant',
     'pes',
     'cam',
     'reflex',
     'objet',
     'medi',
     'pes',
     'adapt',
     'cualqui',
     'siti',
     'pued',
     'coloc',
     'sup',
     'recomendable.d',
     'preis',
     'war',
     'via',
     'germany',
     'günstig',
     'und',
     'die',
     'belieferung',
     'trotzdem',
     'sehr',
     'schnell',
     'werd',
     'wied',
     'bei',
     'ihnen',
     'kauf',
     'und',
     'soi',
     'weiterempfehl',
     'compr',
     'noki',
     '5800',
     'bat',
     'original',
     'apen',
     'dur',
     'unas',
     'hor',
     'tras',
     'años',
     'funcion',
     'bat',
     'dur',
     'sorprendent',
     'dia',
     'aunqu',
     'dud',
     'si',
     'bat',
     'dur',
     'tan',
     'propi',
     'movil',
     'fund',
     'bat',
     'tan',
     'rapidamente.h',
     '10',
     'metr',
     'funcion',
     'trat',
     'postvent',
     'sid',
     'horribl',
     'ped',
     'devolv',
     'dos',
     'product',
     'compr',
     'dand',
     'vuelt',
     'dias',
     'final',
     'permit',
     'devolu',
     'dad',
     'ningun',
     'facil',
     'explic',
     'hac',
     'devolu',
     'envi',
     'reembols',
     'ahor',
     'hac',
     'carg',
     'aconsej',
     'compr',
     'vendedor',
     'devolv',
     'perd',
     'it',
     'about',
     '10',
     'meters',
     'not',
     'working',
     'the',
     'sal',
     'deal',
     'been',
     'horribl',
     'i',
     'asked',
     'return',
     'one',
     'of',
     'the',
     'two',
     'products',
     'i',
     'bought',
     'them',
     'and',
     'i',
     'wer',
     'wandering',
     'around',
     'for',
     'days',
     'in',
     'the',
     'end',
     'allow',
     'the',
     'return',
     'but',
     'i',
     'hav',
     'not',
     'giv',
     'any',
     'explanation',
     'of',
     'how',
     'easily',
     'or',
     'mak',
     'refund',
     'i',
     'had',
     'to',
     'send',
     'cod',
     'and',
     'do',
     'not',
     'tak',
     'car',
     'now',
     'i',
     'advis',
     'you',
     'not',
     'to',
     'buy',
     'anything',
     'from',
     'the',
     'sell',
     'as',
     'you',
     'hav',
     'to',
     'return',
     'thes',
     'lost',
     '.el',
     'prim',
     'contact',
     'maquinill',
     'afeit',
     'buen',
     'robust',
     'buen',
     ...]



Vemos que existen algunos errores:
- Hay tokens formados por dos pablaras unidas por un '.' como por ejemplo 'raton.el'. Deberíamos sustituir el '.' por un especio en blanco y volver a realizar el proceso.
- Tenemos palabras en otros idiomas como por ejemplo: 'wandering', 'days',... Deberíamos hacer una selección del lenguaje. Podríamos comparar nuestras palabras con los paquetes en español de langdetect, langid y Textblob y solo seleccionar los comentarios que se reconocieses como Español. Este proceso no lo vamos a realizar ya que tarda demasiado, y el número de comentarios en otros idiomas no parece significativo como para alterar el resultado.
- Tenemos también algunos tokens que son números, pero que debido a que tienen algún simbolo o letra "pegado" no se han eliminado como por ejemplo '3d', '1-2', ... Tendríamos que ver cómo eliminarlo.

Por lo demás, parece que se ha realizado correctamente.


```python
# Lanzamos el entrenamiento del vectorizer pero con los arrays para mayor velocidad de procesado:
vectorizer.fit(comentarios_train)
```




    CountVectorizer(analyzer='word', binary=False, decode_error='strict',
                    dtype=<class 'numpy.int64'>, encoding='utf-8', input='content',
                    lowercase=True, max_df=1.0, max_features=None, min_df=1,
                    ngram_range=(1, 1), preprocessor=None,
                    stop_words=['de', 'la', 'que', 'el', 'en', 'y', 'a', 'los',
                                'del', 'se', 'las', 'por', 'un', 'para', 'con',
                                'no', 'una', 'su', 'al', 'lo', 'como', 'más',
                                'pero', 'sus', 'le', 'ya', 'o', 'este', 'sí',
                                'porque', ...],
                    strip_accents=None, token_pattern='(?u)\\b\\w\\w+\\b',
                    tokenizer=<function tokear at 0x000002D18BFF09D8>,
                    vocabulary=None)




```python
# Realizamos el transform de los comentarios con el set de train y con el de test
Comentarios_train = vectorizer.transform(comentarios_train)
Comentarios_test  = vectorizer.transform(comentarios_test)

# Y vemos como ha quedado:
print(Comentarios_train)
print(Comentarios_test)
```

      (0, 934)	1
      (0, 1230)	1
      (0, 22046)	1
      (0, 23316)	1
      (0, 34586)	1
      (0, 43261)	1
      (0, 46684)	1
      (0, 51732)	1
      (0, 77273)	1
      (0, 94896)	1
      (0, 120773)	1
      (0, 131478)	1
      (0, 133160)	2
      (0, 134725)	1
      (0, 137592)	1
      (0, 138304)	1
      (0, 143894)	1
      (0, 145140)	1
      (0, 157274)	1
      (0, 163170)	1
      (0, 181491)	1
      (0, 182341)	1
      (0, 183261)	1
      (1, 5905)	1
      (1, 7144)	1
      :	:
      (526832, 23316)	2
      (526832, 26433)	1
      (526832, 29286)	1
      (526832, 33232)	1
      (526832, 43357)	1
      (526832, 54280)	1
      (526832, 67666)	1
      (526832, 118103)	1
      (526832, 140158)	1
      (526832, 141549)	1
      (526832, 144761)	1
      (526832, 161704)	1
      (526832, 173299)	1
      (526833, 19159)	1
      (526833, 39953)	1
      (526833, 67792)	1
      (526833, 86835)	1
      (526833, 120773)	1
      (526833, 132578)	1
      (526833, 144818)	1
      (526833, 146296)	1
      (526833, 152214)	2
      (526833, 160932)	1
      (526833, 173846)	1
      (526833, 182960)	1
      (0, 9792)	2
      (0, 14061)	1
      (0, 17365)	1
      (0, 23316)	1
      (0, 25642)	1
      (0, 27153)	1
      (0, 35450)	1
      (0, 45033)	1
      (0, 51056)	1
      (0, 54597)	1
      (0, 55668)	1
      (0, 63464)	2
      (0, 69555)	1
      (0, 70871)	1
      (0, 78592)	1
      (0, 104902)	1
      (0, 108425)	1
      (0, 112646)	1
      (0, 116257)	1
      (0, 120695)	1
      (0, 132578)	1
      (0, 133884)	1
      (0, 140158)	1
      (0, 144818)	1
      (0, 153597)	1
      :	:
      (175610, 24992)	1
      (175610, 51316)	1
      (175610, 61922)	1
      (175610, 125549)	1
      (175610, 130894)	1
      (175610, 157440)	1
      (175610, 181222)	1
      (175610, 184196)	1
      (175611, 1230)	1
      (175611, 10686)	1
      (175611, 23316)	1
      (175611, 25642)	1
      (175611, 27778)	1
      (175611, 29269)	1
      (175611, 37207)	1
      (175611, 71389)	1
      (175611, 104105)	1
      (175611, 128985)	1
      (175611, 138163)	1
      (175611, 140158)	1
      (175611, 143894)	1
      (175611, 146915)	1
      (175611, 151888)	1
      (175611, 153766)	1
      (175611, 162658)	1
    

Ya tenemos vectorizados los comentarios.


```python
# Con el tiempo que nos ha llevado, vamos a guardar los arrays en un pickle:
import pickle
filename = 'Comentarios_train.pickle'
with open(filename, 'wb') as filehandler:
    pickle.dump(Comentarios_train, filehandler)
    
filename2 = 'Comentarios_test.pickle'
with open(filename2, 'wb') as filehandler:
    pickle.dump(Comentarios_test, filehandler)
```


```python
# Volvemos a cargar los archivos:
Comentarios_train=np.load('C:\\Users\\hesca\\Documents\\MASTER DATAHACK\\Machine Learning\\Practicas\\Comentarios_train.pickle', 
                          allow_pickle=True)
Comentarios_test=np.load('C:\\Users\\hesca\\Documents\\MASTER DATAHACK\\Machine Learning\\Practicas\\Comentarios_test.pickle', 
                          allow_pickle=True)
```

**Vamos a probar distintos modelos.**

## 1.- Regresión Logística


```python
# Importamos las librerías necesarias para este proceso:
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression 
from sklearn.model_selection import GridSearchCV 
```


```python
reglog = LogisticRegression() # Solo un paso, no hace falta pipeline

grid_hyper_reglog ={} # En este caso vacío.

gs_reglog = GridSearchCV(reglog, 
                        param_grid = grid_hyper_reglog,
                        cv=10, # hacemos la validación mediante cross validation
                        scoring = 'roc_auc', 
                        n_jobs=-1,
                        verbose=3) 
```


```python
# lanzamos el entrenamiento del modelo:
gs_reglog.fit(Comentarios_train, notas_train) 
```

    Fitting 10 folds for each of 1 candidates, totalling 10 fits
    

    [Parallel(n_jobs=-1)]: Using backend LokyBackend with 8 concurrent workers.
    [Parallel(n_jobs=-1)]: Done   3 out of  10 | elapsed: 14.0min remaining: 32.6min
    [Parallel(n_jobs=-1)]: Done   7 out of  10 | elapsed: 15.7min remaining:  6.7min
    [Parallel(n_jobs=-1)]: Done  10 out of  10 | elapsed: 18.5min finished
    C:\Users\hesca\Anaconda3\lib\site-packages\sklearn\linear_model\logistic.py:432: FutureWarning: Default solver will be changed to 'lbfgs' in 0.22. Specify a solver to silence this warning.
      FutureWarning)
    




    GridSearchCV(cv=10, error_score='raise-deprecating',
                 estimator=LogisticRegression(C=1.0, class_weight=None, dual=False,
                                              fit_intercept=True,
                                              intercept_scaling=1, l1_ratio=None,
                                              max_iter=100, multi_class='warn',
                                              n_jobs=None, penalty='l2',
                                              random_state=None, solver='warn',
                                              tol=0.0001, verbose=0,
                                              warm_start=False),
                 iid='warn', n_jobs=-1, param_grid={}, pre_dispatch='2*n_jobs',
                 refit=True, return_train_score=False, scoring='roc_auc',
                 verbose=3)




```python
# Para ver el acierto de nuestro modelo con crossvalidation:
gs_reglog.best_score_
```




    0.8808200244041702



## 2.- Árboles de decisión 


```python
# Importamos la librería necesaria:
from sklearn.tree import DecisionTreeClassifier
```


```python
# Definimos pipeline:
arbolito = DecisionTreeClassifier()
# Definimos hiperparámetros:
gryd_hyper_arbolito = {"max_depth": [ 4, 5, 7, 8, 9]}  

gs_arbolito = GridSearchCV(arbolito,
                          param_grid = gryd_hyper_arbolito,
                          cv=10,
                          scoring='roc_auc',
                          n_jobs=-1,
                          verbose=3)
```


```python
# lanzamos el entrenamiento del modelo:
gs_arbolito.fit(Comentarios_train, notas_train) 
```

    Fitting 10 folds for each of 5 candidates, totalling 50 fits
    

    [Parallel(n_jobs=-1)]: Using backend LokyBackend with 8 concurrent workers.
    [Parallel(n_jobs=-1)]: Done  16 tasks      | elapsed:  8.5min
    [Parallel(n_jobs=-1)]: Done  50 out of  50 | elapsed: 30.2min finished
    




    GridSearchCV(cv=10, error_score='raise-deprecating',
                 estimator=DecisionTreeClassifier(class_weight=None,
                                                  criterion='gini', max_depth=None,
                                                  max_features=None,
                                                  max_leaf_nodes=None,
                                                  min_impurity_decrease=0.0,
                                                  min_impurity_split=None,
                                                  min_samples_leaf=1,
                                                  min_samples_split=2,
                                                  min_weight_fraction_leaf=0.0,
                                                  presort=False, random_state=None,
                                                  splitter='best'),
                 iid='warn', n_jobs=-1, param_grid={'max_depth': [4, 5, 7, 8, 9]},
                 pre_dispatch='2*n_jobs', refit=True, return_train_score=False,
                 scoring='roc_auc', verbose=3)




```python
# Vemos el acierto de nuestro modelo:
gs_arbolito.best_score_
```




    0.7371649075581919




```python
# Vemos cual es la mejor profundidad:
gs_arbolito.best_params_
```




    {'max_depth': 9}



Vemos que la mejor profundidad es 9 

## 3.- K-nearest neighbors


```python
# Importamos librerías
from sklearn.neighbors import KNeighborsClassifier
```


```python
vecinos = KNeighborsClassifier()

gryd_hyper_vecinos = {"n_neighbors": [3]}  
# Hacemos prueba solo con 3 vecinos, ya que con más parámetros se eterniza el proceso.

gs_vecinos = GridSearchCV(vecinos,
                          param_grid = gryd_hyper_vecinos,
                          cv=10,
                          scoring='roc_auc',
                          n_jobs=-1,
                          verbose=3)
```


```python
gs_vecinos.fit(Comentarios_train, notas_train) 
```

    Fitting 10 folds for each of 1 candidates, totalling 10 fits
    

    [Parallel(n_jobs=-1)]: Using backend LokyBackend with 8 concurrent workers.
    [Parallel(n_jobs=-1)]: Done   3 out of  10 | elapsed: 95.0min remaining: 221.6min
    [Parallel(n_jobs=-1)]: Done   7 out of  10 | elapsed: 98.7min remaining: 42.3min
    [Parallel(n_jobs=-1)]: Done  10 out of  10 | elapsed: 112.5min finished
    




    GridSearchCV(cv=10, error_score='raise-deprecating',
                 estimator=KNeighborsClassifier(algorithm='auto', leaf_size=30,
                                                metric='minkowski',
                                                metric_params=None, n_jobs=None,
                                                n_neighbors=5, p=2,
                                                weights='uniform'),
                 iid='warn', n_jobs=-1, param_grid={'n_neighbors': [3]},
                 pre_dispatch='2*n_jobs', refit=True, return_train_score=False,
                 scoring='roc_auc', verbose=3)




```python
gs_vecinos.best_params_
```




    {'n_neighbors': 3}



Evidentemente, ya que no hemos probado más parámetros.


```python
gs_vecinos.best_score_
```




    0.6939910533080703



Con el tiempo que nos ha llevado cada modelo, y por la salud del pc, vamos a quedarnos con estos 3 modelos:
    1. Regresión Logística: 0.8808200244041702
    2. Árboles de decisión: 0.7371649075581919
    3. K-Nearest Neoghbors: 0.6939910533080703
        
**Como podemos observar, la Regresión logística nos da el mejor resultado con 0.8808200244041702**

## Probamos el modelo seleccionado con el conjunto test



```python
mejor_modelo = gs_reglog.best_estimator_

mejor_modelo
```




    LogisticRegression(C=1.0, class_weight=None, dual=False, fit_intercept=True,
                       intercept_scaling=1, l1_ratio=None, max_iter=100,
                       multi_class='warn', n_jobs=None, penalty='l2',
                       random_state=None, solver='warn', tol=0.0001, verbose=0,
                       warm_start=False)




```python
mejor_modelo.fit(Comentarios_train, notas_train)
```




    LogisticRegression(C=1.0, class_weight=None, dual=False, fit_intercept=True,
                       intercept_scaling=1, l1_ratio=None, max_iter=100,
                       multi_class='warn', n_jobs=None, penalty='l2',
                       random_state=None, solver='warn', tol=0.0001, verbose=0,
                       warm_start=False)




```python
predicciones_test = mejor_modelo.predict(Comentarios_test)

predicciones_test
```




    array([1, 0, 1, ..., 0, 1, 0], dtype=int64)



**Y ahora vemos que resultados tenemos con distintas métricas**

### AUC-ROC


```python
from sklearn.metrics import roc_auc_score, confusion_matrix, classification_report, f1_score

Score_auc = roc_auc_score(y_true = predicciones_test, y_score = notas_test)

print(Score_auc)
# Vemos el auc_score con el test
```

    0.8061484556357356
    

## F1-Score


```python
F1_score = f1_score(y_true = predicciones_test, y_pred = notas_test)
print(F1_score)
# Vemos el F1_score con el test
```

    0.8141469119206095
    

## Matriz de Confusión


```python
matriz_confusion = confusion_matrix(y_true = predicciones_test, y_pred = notas_test)
                                
matriz_confusion
```




    array([[67110, 16546],
           [17464, 74492]], dtype=int64)




```python
matriz_confusion_df = pd.DataFrame(matriz_confusion)
label = ['positivo', 'negativo']
       
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
      <th>positivo</th>
      <th>negativo</th>
    </tr>
    <tr>
      <th>Real</th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>positivo</td>
      <td>67110</td>
      <td>16546</td>
    </tr>
    <tr>
      <td>negativo</td>
      <td>17464</td>
      <td>74492</td>
    </tr>
  </tbody>
</table>
</div>




```python
import matplotlib.pyplot as plt
%matplotlib inline
import seaborn as sns
plt.figure(figsize=(8,4))
sns.heatmap(matriz_confusion_df,                     
            annot=True,
            fmt="d",
            cmap="Blues")
pass
```


![png](output_58_0.png)


## Probamos nuestro modelo con unos comentarios que me invento

Por probar un poco más nuestro modelo, vamos a meter una serie de comentarios inventados por mi y una valoración de los mismo, a ver si conseguimos que nuestro modelo funcione.


```python
comentarios_prueba = np.array([("Qué gran producto, estoy realmente satisfecho."),
                      ("Es un producto bueno en relación calidad precio"), 
                      ("No me convence el producto, creo que lo voy a devolver"), 
                      ("Es un producto terrible, no lo recomiendo para nada"),
                      ("No me terminan de agradar, estarían mejor si no fueran tan verdes"),
                      ])

notas_prueba = np.array([1, 1, 0, 0, 0])
```


```python
# Lanzamos el transform de nuestra prueba:
Comentarios_prueba = vectorizer.transform(comentarios_prueba)

print(Comentarios_prueba)
```

      (0, 81230)	1
      (0, 142724)	1
      (0, 148949)	1
      (0, 158774)	1
      (1, 23316)	1
      (1, 25642)	1
      (1, 140158)	1
      (1, 142724)	1
      (1, 151888)	1
      (2, 39313)	1
      (2, 41403)	1
      (2, 50444)	1
      (2, 142724)	1
      (2, 187554)	1
      (3, 142724)	1
      (3, 149737)	1
      (3, 174604)	1
      (4, 4061)	1
      (4, 111903)	1
      (4, 162658)	1
      (4, 172049)	1
      (4, 174377)	1
      (4, 185010)	1
    


```python
predicciones_prueba = mejor_modelo.predict(Comentarios_prueba)

predicciones_prueba
```




    array([1, 1, 0, 0, 0], dtype=int64)



# Vemos el F1_score


```python
F1_score = f1_score(y_true = predicciones_prueba, y_pred = notas_prueba)

print(F1_score)
```

    1.0
    

Realmente con los comentarios que he introducido era previsible que el modelo acertara a la perfección, pero quería comprobar que el modelo podría funcionar con otros comentarios.
