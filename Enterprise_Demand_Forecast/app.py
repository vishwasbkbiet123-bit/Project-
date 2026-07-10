from flask import Flask, request, render_template
import pickle
import numpy as np

app = Flask(__name__)

# Load the saved Model and Encoder
model_path = 'models/rf_demand_model.pkl'
encoder_path = 'models/label_encoder.pkl'

with open(model_path, 'rb') as file:
    model = pickle.load(file)

with open(encoder_path, 'rb') as file:
    encoder = pickle.load(file)

@app.route('/')
def home():
    # Encoder ke andar se saare product names (classes) nikal rahe hain
    # aur unhe alphabetically sort kar rahe hain taaki dropdown accha dikhe
    product_list = list(encoder.classes_)
    product_list.sort()
    
    # render_template ke zariye yeh list index.html me bhej rahe hain
    return render_template('index.html', products=product_list)

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        try:
            # HTML Form se data lena (sab string format mein aata hai, isliye int mein convert karna zaroori hai)
            product_name = request.form['description']
            month = int(request.form['month'])
            day_of_week = int(request.form['day_of_week'])
            is_weekend = int(request.form['is_weekend'])
            is_seasonal = int(request.form['is_seasonal'])
            
            # Product name ko numeric ID mein convert karna
            try:
                product_id = encoder.transform([product_name])[0]
            except ValueError:
                return render_template('result.html', error="Product not recognized by the model. Please check the spelling.")
            
            # Model input format tayar karna
            features = np.array([[product_id, month, day_of_week, is_weekend, is_seasonal]])
            
            # Prediction
            predicted_quantity = model.predict(features)[0]
            final_demand = int(round(predicted_quantity))
            
            # Result page par variables bhejna
            return render_template('result.html', product=product_name, demand=final_demand)
            
        except Exception as e:
            return render_template('result.html', error=str(e))

if __name__ == '__main__':
    app.run(debug=True, port=5000)