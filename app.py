import pandas as pd
import pickle
from flask import Flask, request, jsonify, render_template

app = Flask(__name__)

# Load the trained model, scaler, and encoder
with open('best_model.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

with open('scaler.pkl', 'rb') as scaler_file:
    scaler = pickle.load(scaler_file)

with open('onehot_encoder.pkl', 'rb') as encoder_file:
    onehot_encoder = pickle.load(encoder_file)

# Define the numerical and categorical features
numerical_features = ['Order Quantity', 'Unit Price ']
categorical_features = ['Category', 'Delivery Location', 'Product Detail (type, material, color, size)', 'Price Category', 'Customer Name', 'Gender']

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.form.to_dict()
    
    # Convert data types
    data['Order Quantity'] = int(data['Order Quantity'])
    data['Unit Price '] = float(data['Unit Price '])
    
    # Convert the form data to a DataFrame
    df = pd.DataFrame([data])
    
    # Process numerical and categorical features
    numerical_data = scaler.transform(df[numerical_features])
    categorical_data = onehot_encoder.transform(df[categorical_features])
    
    # Combine the processed numerical and categorical features
    processed_data = pd.DataFrame(numerical_data, columns=numerical_features).join(
        pd.DataFrame(categorical_data, columns=onehot_encoder.get_feature_names_out(categorical_features))
    )
    
    # Make a prediction
    prediction = model.predict(processed_data)
    
    return render_template('index.html', prediction=prediction[0])

if __name__ == '__main__':
    app.run(debug=True)