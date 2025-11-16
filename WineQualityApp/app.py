from flask import Flask, request, render_template
import numpy as np
import pandas as pd
import joblib

# Load model
model = joblib.load("wine_rf_model.pkl")

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    data = [
        float(request.form['fixed_acidity']),
        float(request.form['volatile_acidity']),
        float(request.form['citric_acid']),
        float(request.form['residual_sugar']),
        float(request.form['chlorides']),
        float(request.form['free_sulfur_dioxide']),
        float(request.form['total_sulfur_dioxide']),
        float(request.form['density']),
        float(request.form['pH']),
        float(request.form['sulphates']),
        float(request.form['alcohol'])
    ]

    input_array = np.array(data).reshape(1,-1)
    prediction = model.predict(input_array)[0]
    prediction = round(prediction)

    return render_template('index.html', prediction_text=f"Predicted Quality: {prediction}")

if __name__ == "__main__":
    app.run(debug=True)
