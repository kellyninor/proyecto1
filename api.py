from flask import Flask, request, jsonify
import joblib
import pandas as pd

# Cargar el modelo entrenado y el preprocesador
loaded_model = joblib.load('random_forest_model.pkl')
preprocessor = joblib.load('preprocessor.pkl')
scaler = joblib.load('scaler.pkl')

# Inicializar la aplicación Flask
app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict_price():
    data = request.get_json()  # Obtener datos de la solicitud POST
    year = data['Year']
    mileage = data['Mileage']
    state = data['State']
    make = data['Make']
    model = data['Model']

    
    # Crear un DataFrame con los datos de entrada
    input_data = pd.DataFrame({
        'Year': [year],
        'Mileage': [mileage],
        'State': [state],
        'Make': [make],
        'Model': [model]
    })

    input_data['Age']= datetime.datetime.now().year - input_data['Year']

    # Preprocesar los datos de entrada
    processed_input_data = preprocessor.transform(input_data)
    
    # Hacer predicciones utilizando el modelo cargado
    predictions = loaded_model.predict(processed_input_data)

    # Invertir la transformación para obtener el precio predicho
    predictions_new = predictions.reshape(-1, 1)
    predicted_price = scaler.inverse_transform(predictions_new).flatten()
    
    # Crear un diccionario con los resultados
    results = {'Price': float(predicted_price)}
    
    return jsonify(results)

if __name__ == '__main__':
    app.run(debug=True, use_reloader=False, host='0.0.0.0', port=5000)