from flask import Flask, render_template, request, jsonify
import joblib
import numpy as np
import pandas as pd

app = Flask(__name__)

# Load model, scaler, dan nama fitur
model = joblib.load('model_logistic.joblib')
scaler = joblib.load('scaler.joblib')
feature_names = joblib.load('feature_names.joblib')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.form.to_dict()
        
        # Konversi input pengguna ke DataFrame sesuai nama fitur
        input_data = []
        for feature in feature_names:
            val = float(data.get(feature, 0))
            input_data.append(val)
            
        input_df = pd.DataFrame([input_data], columns=feature_names)
        
        # Scaling input
        input_scaled = scaler.transform(input_df)
        
        # Prediksi & Probabilitas
        prediction = model.predict(input_scaled)[0]
        probability = model.predict_proba(input_scaled)[0][1] # Probabilitas Default
        
        result = {
            'status': 'success',
            'prediction': int(prediction),
            'risk_label': 'Berisiko Gagal Bayar' if prediction == 1 else 'Tidak Berisiko',
            'probability': round(float(probability) * 100, 2),
            'risk_level': 'Tinggi' if probability > 0.5 else ('Sedang' if probability > 0.3 else 'Rendah')
        }
        return jsonify(result)
    
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 400

if __name__ == '__main__':
    app.run(debug=True)