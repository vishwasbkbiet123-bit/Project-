from flask import Flask, render_template, request
import joblib
import numpy as np

app = Flask(__name__)

# Load the saved model and scaler
model = joblib.load('models/model.pkl')
scaler = joblib.load('models/scaler.pkl')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        try:
            # Get numerical values from the HTML form
            discounted_price = float(request.form['discounted_price'])
            actual_price = float(request.form['actual_price'])
            discount_percentage = float(request.form['discount_percentage']) / 100.0
            rating = float(request.form['rating'])
            main_category = request.form['main_category']

            # Create One-Hot Encoding EXACTLY as the training dataset generated
            # (Computers&Accessories was dropped by drop_first=True, so it means all zeros)
            is_electronics = 1 if main_category == 'Electronics' else 0
            is_home_kitchen = 1 if main_category == 'Home&Kitchen' else 0
            is_musical = 1 if main_category == 'MusicalInstruments' else 0
            is_office = 1 if main_category == 'OfficeProducts' else 0
            is_other = 1 if main_category == 'Other' else 0

            # Total 9 features (4 numeric + 5 categorical dummies)
            features = [
                discounted_price, 
                actual_price, 
                discount_percentage, 
                rating,
                is_electronics,
                is_home_kitchen,
                is_musical,
                is_office,
                is_other
            ]
            
            final_features = np.array(features).reshape(1, -1)

            # Scale the features using the saved scaler
            scaled_features = scaler.transform(final_features)

            # Predict the demand
            log_prediction = model.predict(scaled_features)
            
            # Convert log scale back to actual numbers
            actual_prediction = int(np.expm1(log_prediction)[0])

            return render_template('result.html', prediction_text=f'Predicted Demand (Rating Count): {actual_prediction} units')

        except Exception as e:
            return render_template('result.html', prediction_text=f'Error in prediction: {str(e)}')

if __name__ == "__main__":
    app.run(debug=True)