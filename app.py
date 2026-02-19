# Save this code as app.py

import os
from flask import Flask, request, jsonify
import joblib
import pandas as pd
import numpy as np

app = Flask(__name__)

# Global variables to store loaded models and preprocessing artifacts
kmeans_model = None
rf_model = None
xgb_model = None
scaler = None
model_columns = None
luxury_threshold = None

def load_models_and_artifacts():
    global kmeans_model, rf_model, xgb_model, scaler, model_columns, luxury_threshold
    try:
        kmeans_model = joblib.load('kmeans_model.joblib')
        rf_model = joblib.load('random_forest_model.joblib')
        xgb_model = joblib.load('xgboost_model.joblib')
        scaler = joblib.load('scaler.joblib')
        model_columns = joblib.load('model_columns.joblib')
        # Load the raw dataset to calculate luxury_threshold dynamically
        # In a real-world scenario, luxury_threshold would be saved as an artifact
        # For this example, we re-calculate it from the original dataset for consistency
        df_raw = pd.read_csv('Travel details dataset.csv')
        for col in ['Accommodation cost', 'Transportation cost']:
            df_raw[col] = df_raw[col].astype(str).str.replace(r'[^A-z0-9.]', '', regex=True)
            df_raw[col] = pd.to_numeric(df_raw[col], errors='coerce')
            df_raw[col] = df_raw[col].fillna(df_raw[col].median())
        df_raw['Total_Cost'] = df_raw['Accommodation cost'] + df_raw['Transportation cost']
        df_raw['Duration (days)'] = df_raw['Duration (days)'].fillna(df_raw['Duration (days)'].median())
        df_raw['Cost_per_Day'] = df_raw['Total_Cost'] / df_raw['Duration (days)']
        luxury_threshold = df_raw['Cost_per_Day'].quantile(0.75)

        print('All models and artifacts loaded successfully.')
    except Exception as e:
        print(f'Error loading models or artifacts: {e}')
        exit(1)

# Preprocessing function
def preprocess_input(data):
    global luxury_threshold # Access the global luxury_threshold

    # Convert input dict to DataFrame
    df_input = pd.DataFrame([data])

    # 1. Clean 'Accommodation cost' and 'Transportation cost'
    for col in ['Accommodation cost', 'Transportation cost']:
        if col in df_input.columns:
            df_input[col] = df_input[col].astype(str).str.replace(r'[^A-z0-9.]', '', regex=True)
            df_input[col] = pd.to_numeric(df_input[col], errors='coerce')
            # For new data, if missing, fill with a sensible default or median from training data
            # For this example, we'll assume valid input for simplicity, or re-use previous median
            # In a real API, you'd load the median as an artifact.
            if df_input[col].isnull().any():
                # Fallback if original median was not explicitly saved
                df_input[col] = df_input[col].fillna(900.0 if col == 'Accommodation cost' else 550.0) # Example fallback

    # 2. Impute numerical columns if needed (Duration (days), Traveler age)
    for col in ['Duration (days)', 'Traveler age']:
        if col in df_input.columns and df_input[col].isnull().any():
            # Fallback if original median was not explicitly saved
            df_input[col] = df_input[col].fillna(7.0 if col == 'Duration (days)' else 31.0) # Example fallback

    # 3. Impute categorical columns with mode if needed
    categorical_cols_for_imputation = ['Destination', 'Traveler gender', 'Traveler nationality', 'Accommodation type', 'Transportation type']
    for col in categorical_cols_for_imputation:
        if col in df_input.columns and df_input[col].isnull().any():
            # Fallback if original mode was not explicitly saved
            if col == 'Destination': mode_val = 'Bali'
            elif col == 'Traveler gender': mode_val = 'Female'
            elif col == 'Traveler nationality': mode_val = 'American'
            elif col == 'Accommodation type': mode_val = 'Hotel'
            elif col == 'Transportation type': mode_val = 'Plane'
            else: mode_val = None # Should not happen with defined cols
            df_input[col] = df_input[col].fillna(mode_val)

    # 4. Process date columns
    df_input['Start date'] = pd.to_datetime(df_input['Start date'], errors='coerce')
    df_input['Start_Month'] = df_input['Start date'].dt.month
    df_input['Start_Day_of_Week'] = df_input['Start date'].dt.dayofweek
    
    # Impute missing Start_Month/Start_Day_of_Week if date conversion failed (e.g., malformed date string)
    if df_input['Start_Month'].isnull().any():
        df_input['Start_Month'] = df_input['Start_Month'].fillna(7.0) # Example fallback median from training
    if df_input['Start_Day_of_Week'].isnull().any():
        df_input['Start_Day_of_Week'] = df_input['Start_Day_of_Week'].fillna(3.0) # Example fallback median from training

    # 5. Create new features 'Total_Cost' and 'Cost_per_Day' (Total_Cost will be dropped later)
    df_input['Total_Cost'] = df_input['Accommodation cost'] + df_input['Transportation cost']
    df_input['Cost_per_Day'] = df_input['Total_Cost'] / df_input['Duration (days)']
    # Impute Cost_per_Day if division by zero or NaN occurred
    if df_input['Cost_per_Day'].isnull().any():
        df_input['Cost_per_Day'] = df_input['Cost_per_Day'].fillna(200.0) # Example fallback median

    # 6. Apply One-Hot Encoding
    # Ensure consistent columns using model_columns
    categorical_cols_for_ohe = ['Destination', 'Traveler gender', 'Traveler nationality', 'Accommodation type', 'Transportation type']
    df_encoded = pd.get_dummies(df_input, columns=categorical_cols_for_ohe, drop_first=True)

    # Align columns with training data using model_columns
    df_processed = pd.DataFrame(columns=model_columns)
    for col in model_columns:
        if col in df_encoded.columns:
            df_processed[col] = df_encoded[col]
        else:
            # Add missing columns with default value 0 (for one-hot encoded features)
            df_processed[col] = 0
    
    # Ensure correct data types (especially for boolean columns becoming 0/1)
    for col in df_processed.columns:
        if df_processed[col].dtype == 'bool':
            df_processed[col] = df_processed[col].astype(int)

    # 7. Add K-Means Cluster Assignment
    # The scaler was fitted on X_train WITHOUT the 'Cluster' column yet.
    # So we need to scale the current features and then predict the cluster.
    # Drop 'Cluster', 'Is_Luxury_Trip', 'Is_Long_Trip' if they are in model_columns to scale only original features
    # And re-add them after scaling and cluster prediction.

    # Filter to only the features that were originally scaled before K-Means prediction
    # These are all columns in model_columns except 'Cluster', 'Is_Luxury_Trip', 'Is_Long_Trip' and the OHE features that were originally in X_train
    # Need to reconstruct the original X_train columns (numerical + one-hot encoded) before 'Cluster', 'Is_Luxury_Trip', 'Is_Long_Trip' were added

    # Find the index where 'Cluster' or 'Is_Luxury_Trip' might have been inserted
    # Assuming model_columns was saved BEFORE 'Cluster' was added to X_train for KMeans scaling
    # The scaler was trained on X_train before 'Cluster' was added.
    # So, we need to create the feature set that matches what the scaler expects.

    # Identify columns that were part of X_train before 'Cluster' and rule-based features were added
    features_for_scaling = [col for col in model_columns if col not in ['Cluster', 'Is_Luxury_Trip', 'Is_Long_Trip']]
    
    # Create a DataFrame with only the columns that the scaler expects
    df_scaled_features = df_processed[features_for_scaling]

    # Scale features
    X_scaled = scaler.transform(df_scaled_features)

    # Predict cluster for the input data
    cluster_assignment = kmeans_model.predict(X_scaled)[0]
    df_processed['Cluster'] = cluster_assignment
    
    # 8. Add Rule Engine Features
    df_processed['Is_Luxury_Trip'] = (df_processed['Cost_per_Day'] > luxury_threshold).astype(int)
    df_processed['Is_Long_Trip'] = (df_processed['Duration (days)'] > 10).astype(int)

    # Ensure the final DataFrame has columns in the exact order as model_columns
    final_features = df_processed[model_columns]

    return final_features

@app.route('/predict', methods=['POST'])
def predict():
    if not request.json:
        return jsonify({'error': 'Please send JSON data'}), 400

    data = request.json

    try:
        processed_data = preprocess_input(data)

        # Make predictions from both models
        rf_prediction = rf_model.predict(processed_data)[0]
        xgb_prediction = xgb_model.predict(processed_data)[0]

        # Ensemble prediction by averaging
        ensemble_prediction = (rf_prediction + xgb_prediction) / 2

        return jsonify({'predicted_total_cost': round(ensemble_prediction, 2)}),

    except KeyError as e:
        return jsonify({'error': f'Missing expected input data: {e}'}), 400
    except Exception as e:
        return jsonify({'error': f'Prediction failed: {e}'}), 500


if __name__ == '__main__':
    load_models_and_artifacts()
    app.run(debug=True, host='0.0.0.0', port=5000)
