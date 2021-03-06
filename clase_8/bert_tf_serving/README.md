# Python BERT classifier with Tensorflow Serving
![img1](images/sentiment_analysis.png)\

# Acerca del ejemplo ๐
Pequeรฑo ejemplo de como utilizar TFBert con TF serving utilizando docker y Flask.

# Requirements ๐
| Package     | Version |
| ----------- | ------- |
| Python      | 3.7     |
| Flask       | 2.0     |
| Tensorflow  | 2.6.0   |
| TF serving  | 2.6.0   |
| Transformes | 4.11.3  |
| Docker      |         |

# Exportar su modelo Bert para TF serving ๐งโ๏ธ
Siga las instrucciones en el notebook bert_2_tfx.ipynb para exportar el modelo.

# Docker BERT TFX service ๐งโ๏ธ
Descargar la imagen de TF serving para docker
```
docker pull tensorflow/serving:2.6.0
```

Lanzar un contenedor con el modelo de BERT exportado:
```
docker run --rm -it -p 8501:8501 -v $(pwd)/mybert:/models/mybert -e MODEL_NAME=mybert -t tensorflow/serving:2.6.0
```

Lanzar el contenedor en modo "daemon"
```
docker run --rm -d -p 8501:8501 -v $(pwd)/mybert:/models/mybert -e MODEL_NAME=mybert -t tensorflow/serving:2.6.0
```

Una vez lanzado el contenedor con el modelo, podemos probarlo con test.py

# Flask application up ๐งโ๏ธ
- Abrir visual studio code dentro de la carpeta "app"
- Run app.py script.
- Ir a la URL 127.0.0.1:5000
