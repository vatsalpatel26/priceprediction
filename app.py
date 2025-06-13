from flask import Flask, render_template, request
import pickle
import numpy as np
import os

app = Flask(__name__)

# Get the absolute path to the model file
model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'house_price_prediction.pkl')

try:
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
except Exception as e:
    print(f"Error loading model: {str(e)}")
    model = None

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if model is None:
        return render_template('index.html', prediction_text="Error: Model not loaded properly")
        
    try:
        features = [
            float(request.form['CRIM']),
            float(request.form['ZN']),
            float(request.form['INDUS']),
            float(request.form['CHAS']),
            float(request.form['NOX']),
            float(request.form['RM']),
            float(request.form['AGE']),
            float(request.form['DIS']),
            float(request.form['RAD']),
            float(request.form['TAX']),
            float(request.form['PTRATIO']),
            float(request.form['B']),
            float(request.form['LSTAT'])
        ]
        
        features_array = np.array([features])
        prediction = model.predict(features_array)
        output = round(prediction[0],2)
        
        return render_template('index.html', prediction_text=f"Predicted Price: {output}")
    except Exception as e:
        return render_template('index.html', prediction_text=f"Error making prediction: {str(e)}")


if __name__ == "__main__":
    app.run(debug=True)