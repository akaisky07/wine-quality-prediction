# Import necessary libraries
from flask import Flask, render_template, request
import numpy as np
import joblib

app = Flask(__name__)

# Load the pre-trained model
model = joblib.load('wine_prediction.pkl')

@app.route('/', methods=['GET', 'POST'])
def predict_wine_quality():
    if request.method == 'POST':
        # Get input data from the form
        fixed_acidity = float(request.form['fixed_acidity'])
        volatile_acidity = float(request.form['volatile_acidity'])
        citric_acid = float(request.form['citric_acid'])
        residual_sugar = float(request.form['residual_sugar'])
        chlorides = float(request.form['chlorides'])
        free_sulfur_dioxide = float(request.form['free_sulfur_dioxide'])
        total_sulfur_dioxide = float(request.form['total_sulfur_dioxide'])
        density = float(request.form['density'])
        pH = float(request.form['pH'])
        sulphates = float(request.form['sulphates'])
        alcohol = float(request.form['alcohol'])
        
        # Create a NumPy array with the input data
        input_data = np.array([
            [fixed_acidity, volatile_acidity, citric_acid, residual_sugar, chlorides, 
             free_sulfur_dioxide, total_sulfur_dioxide, density, pH, sulphates, alcohol]
        ])
        
        # Make a prediction using the pre-trained model
        prediction = model.predict(input_data)
        
        # Map the prediction to "Good" (1) or "Bad" (0)
        quality = "Good" if prediction[0] == 1 else "Bad"

        return render_template('result.html', quality=quality)
    
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)

