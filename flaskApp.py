from flask import Flask, request, jsonify
import joblib
import pandas as pd
import Analysis

app = Flask(__name__)

# Load the model
model = joblib.load('regression_model.pkl')

@app.route('/predict', methods=['POST'])
def predict():
    #Parse Json data
    input_data = request.get_json()
    
    # Convert JSON file to DataFrame
    input_df = pd.DataFrame([input_data])
    
    # Predict using the model
    prediction = model.predict(input_df)
    
    # Return prediction
    return jsonify({'prediction': prediction.tolist()})
if __name__ == '__main__':
    app.run(debug=True)