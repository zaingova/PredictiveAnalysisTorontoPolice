from flask import Flask, render_template, request, jsonify
import joblib
import pandas as pd
import os

app = Flask(__name__)

# Load models and feature names
models_dir = os.path.join(os.path.dirname(__file__), 'models')
log_reg_model = joblib.load(os.path.join(models_dir, 'logistic_regression_model.pkl'))
tree_model = joblib.load(os.path.join(models_dir, 'decision_tree_model.pkl'))
feature_names = joblib.load(os.path.join(models_dir, 'feature_names.pkl'))

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get form data
        data = request.json
        primary_offence = data.get('primary_offence')
        location_type = data.get('location_type')
        neighbourhood = data.get('neighbourhood')
        religion_bias = data.get('religion_bias')
        
        # Create a DataFrame with all features initialized to 0
        input_df = pd.DataFrame(0, index=[0], columns=feature_names)
        
        # Set the one-hot encoded features to 1 based on user input
        # Note: drop_first=True was used in encoding, so we need to handle that
        
        if primary_offence != "Not specified":
            feature_name = f'PRIMARY_OFFENCE_{primary_offence}'
            if feature_name in input_df.columns:
                input_df[feature_name] = 1
        
        if location_type != "Not specified":
            feature_name = f'LOCATION_TYPE_{location_type}'
            if feature_name in input_df.columns:
                input_df[feature_name] = 1
        
        if neighbourhood != "Not specified":
            feature_name = f'NEIGHBOURHOOD_158_{neighbourhood}'
            if feature_name in input_df.columns:
                input_df[feature_name] = 1
        
        if religion_bias != "Not specified":
            feature_name = f'RELIGION_BIAS_{religion_bias}'
            if feature_name in input_df.columns:
                input_df[feature_name] = 1
        
        # Make predictions
        log_reg_pred = log_reg_model.predict(input_df)[0]
        log_reg_prob = log_reg_model.predict_proba(input_df)[0]
        
        tree_pred = tree_model.predict(input_df)[0]
        tree_prob = tree_model.predict_proba(input_df)[0]
        
        # Prepare response
        response = {
            'logistic_regression': {
                'prediction': 'Yes' if log_reg_pred == 1 else 'No',
                'probability': float(log_reg_prob[1] * 100)
            },
            'decision_tree': {
                'prediction': 'Yes' if tree_pred == 1 else 'No',
                'probability': float(tree_prob[1] * 100)
            }
        }
        
        return jsonify(response)
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, port=5000)
