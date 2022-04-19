# -------------------------------------------------------------------
# CREAR NUESTRA API con Flask
# ------------------------------------------------------------------
import traceback
from flask import Flask, jsonify, request, render_template
# Crear el server Flask
app = Flask(__name__)

# Crear el modelo que utilizaremos en la API
from mybert import MyBertModel
modelo = MyBertModel()

@app.route("/")
def index():
    return render_template('index.html')

@app.route("/predict", methods=['POST'] )
def predict():
    if request.method == 'POST':
        # Siempre es recomendable colocar nuestro
        # código entre try except para que el servidor
        # no se caiga si llega a fallar algo
        try:
            input_text = str(request.form['input_text'])
            try:
                prediction_text = modelo.predict(input_text)
            except:
                prediction_text = "service is not available at the moment"
            return render_template('index.html', prediction_text=prediction_text)
        except:
            # En caso de falla, retornar el mensaje de error
            return jsonify({'trace': traceback.format_exc()})


if __name__ == '__main__':
    # Lanzar server
    app.run(host="127.0.0.1", port=5000)