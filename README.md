# Procesamiento de lenguaje natural

## Localizacion de los desafíos

- Desafío 1: [Link](https://github.com/gmvtrmaga/procesamiento_lenguaje_natural/blob/main/desafios/desafio_1/Desafio_1.ipynb)
- Desafío 2: [Link](https://github.com/gmvtrmaga/procesamiento_lenguaje_natural/blob/main/desafios/desafio_2/Desafio_2.ipynb)
- Desafío 3.word: [Link](https://github.com/gmvtrmaga/procesamiento_lenguaje_natural/blob/main/desafios/desafio_3/Desafio_3_word.ipynb)
- Desafío 3.char: [Link](https://github.com/gmvtrmaga/procesamiento_lenguaje_natural/blob/main/desafios/desafio_3/Desafio_3_char.ipynb)
- Desafío 4: [Link](https://github.com/gmvtrmaga/procesamiento_lenguaje_natural/blob/main/desafios/desafio_4/Desafio_4.ipynb)
- Desafío 5: [Link](https://github.com/gmvtrmaga/procesamiento_lenguaje_natural/blob/main/desafios/desafio_5/Desafio_5.ipynb)

# Como ejecutar el repositorio:
NOTA: La versión de Python utilizada es la 3.10.0.
La notación de los comandos y los imports de las librerías de las notebooks puede variar ligeramente dependiendo del sistema operativo y de la versión de python que se esté utilizando, asi que se recomienda reproducir este código con la versión de python recomendada.

Para realizar el setup del repositorio, en el directorio principal escribir los siguientes comandos:
- ``py -m venv .env``
- ``.\.env\Scripts\activate``
- ``py -m pip install --upgrade pip``
- ``pip install -r .\requirements.txt``

Una vez instaladas todas las dependencias, realizar los siguientes pasos:
- Abrir los documentos especificados en la sección anterior
- Dentro del notebook seleccionar el kernel de ``.env``
- Ejecutar celdas secuencialmente

## Descripción de los desafíos:

### Desafío 1: Vectorización de texto y modelo de clasificación Naïve Bayes

Este desafío supone una introducción a la materia, en la que se emplean alguna de las herramientas de sklearn para vectorizar documentos y computar la similaridad del contenido de texto. Además se realizaron entrenamientos con modelos de clasificación de ComplementNB y Naive Bayes y como la modificacion de los hiperparámetros de éstos modelos o del vectorizador pueden mejorar el output que proporcionan.

### Desafío 2: Custom embeddings con Gensim

El segundo desafío introduce el uso de embeddings entrenables para la vectorización de un conjunto de texto y realizar varios análisis exploratorios de conjuntos de palabras, como distancia entre palabras o representación de las mismas en 2 o 3 dimensiones

### Desafío 3.word: Predicción de próxima palabra

En este desafío se trabaja con un modelo de procesamiento de lenguaje el cual, a partir de un texto de entrada, es capaz de predecir un conjunto de palabras que sucedería a dicha entrada. Este modelo es capaz de continuar frases con cierta coherencia hasta un número determinado de palabras. En este desafío aparecen nuevos conceptos, como la perplejidad, que permite determinar la calidad de prediccion de un modelo.

### Desafío 3.char: Predicción de próxima palabra

En este desafío se trabaja con un modelo de procesamiento de lenguaje el cual, a partir de un texto de entrada, es capaz de predecir un conjunto de caracteres que sucedería a dicha entrada. Este modelo es capaz de completar palabras incompletas con una precision aceptable. En este desafío aparecen nuevos conceptos, como la perplejidad, que permite determinar la calidad de prediccion de un modelo.

### Desafío 4: LSTM Bot QA

Este desafío pone en práctica el uso de embeddings de palabras y redes neuronales recurrentes para el diseño de un bot capaz de responder preguntas introducidas por el usuario.

### Desafío 5: BERT Sentiment Analysis

En este desafío, se entrenó un modelo de BERT para realizar análisis de sentimientos a partir de un dataset de críticas de Google Apps
