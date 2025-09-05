# app.py - Flask Backend Server
from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
import numpy as np
import joblib
import os
from sklearn.preprocessing import LabelEncoder
import warnings
warnings.filterwarnings('ignore')

app = Flask(__name__)
CORS(app)  # Enable CORS for React frontend

# Global variables to store model and preprocessing info
trained_model = None
feature_columns = None
label_encoders = None

def load_model_and_preprocessors():
    """Load the trained model and preprocessing components"""
    global trained_model, feature_columns, label_encoders
    
    try:
        # Load the saved model
        model_data = joblib.load('sleep_deprivation_model.pkl')
        trained_model = model_data['model']
        feature_columns = model_data['feature_columns']
        label_encoders = model_data['label_encoders']
        print("Model loaded successfully!")
        return True
    except Exception as e:
        print(f"Error loading model: {str(e)}")
        return False

def preprocess_input(input_data):
    """Preprocess input data to match training format"""
    try:
        # Create DataFrame from input
        df = pd.DataFrame([input_data])
        
        # Handle time features
        if 'bedtime' in df.columns:
            def time_to_minutes(time_str):
                try:
                    if pd.isna(time_str) or time_str == '':
                        return 0
                    time_str = str(time_str).strip()
                    if ':' in time_str:
                        hours, minutes = map(int, time_str.split(':'))
                        return hours * 60 + minutes
                    else:
                        return float(time_str)
                except:
                    return 0

            df['bedtime_minutes'] = df['bedtime'].apply(time_to_minutes)
            df = df.drop('bedtime', axis=1)

        if 'wakeup_time' in df.columns:
            df['wakeup_minutes'] = df['wakeup_time'].apply(time_to_minutes)
            df = df.drop('wakeup_time', axis=1)

        # Apply label encoding for categorical variables
        if label_encoders:
            for col in df.columns:
                if col in label_encoders and df[col].dtype == 'object':
                    le = label_encoders[col]
                    try:
                        df[col] = le.transform(df[col].astype(str))
                    except ValueError:
                        # Handle unknown values
                        df[col] = 0

        # Convert object columns to numeric
        for col in df.columns:
            if df[col].dtype == 'object':
                try:
                    df[col] = pd.to_numeric(df[col], errors='coerce')
                except:
                    df[col] = 0

        # Fill NaN values
        df = df.fillna(0)

        # Ensure all columns are numeric
        for col in df.columns:
            if df[col].dtype not in ['int8', 'int16', 'int32', 'int64', 'float16', 'float32', 'float64', 'bool']:
                df[col] = df[col].astype('float64')

        return df
        
    except Exception as e:
        print(f"Error in preprocessing: {str(e)}")
        return None

def create_engineered_features(df):
    """Create engineered features"""
    try:
        df_eng = df.copy()

        # Sleep efficiency features
        if 'sleep_hours' in df_eng.columns and 'sleep_latency' in df_eng.columns:
            df_eng['sleep_efficiency'] = df_eng['sleep_hours'] / (df_eng['sleep_hours'] + df_eng['sleep_latency']/60 + 1e-6)

        # Technology usage intensity
        if 'screen_time_hours' in df_eng.columns and 'social_media_hours' in df_eng.columns:
            df_eng['total_screen_time'] = df_eng['screen_time_hours'] + df_eng['social_media_hours']

        if 'in_bed_phone_use_percent' in df_eng.columns and 'total_screen_time' in df_eng.columns:
            df_eng['digital_dependency'] = (df_eng['in_bed_phone_use_percent'] / 100) * df_eng['total_screen_time']

        # Health score
        health_components = []
        if 'physical_activity_mins' in df_eng.columns:
            health_components.append(df_eng['physical_activity_mins'] / 60)
        if 'water_intake_liters' in df_eng.columns:
            health_components.append(df_eng['water_intake_liters'] / 3)
        if 'time_spent_outdoors_daily' in df_eng.columns:
            health_components.append(df_eng['time_spent_outdoors_daily'] / 120)
        if 'stress_level' in df_eng.columns:
            health_components.append(-df_eng['stress_level'] / 10)
        if 'caffeine_intake' in df_eng.columns:
            health_components.append(-df_eng['caffeine_intake'] / 400)

        if health_components:
            df_eng['health_score'] = sum(health_components)

        # Sleep quality composite
        sleep_quality_components = []
        if 'sleep_quality' in df_eng.columns:
            sleep_quality_components.append(df_eng['sleep_quality'])
        if 'sleep_consistency_score' in df_eng.columns:
            sleep_quality_components.append(df_eng['sleep_consistency_score'])
        if 'sleep_latency' in df_eng.columns:
            sleep_quality_components.append(-df_eng['sleep_latency'] / 30)

        if sleep_quality_components:
            df_eng['sleep_quality_composite'] = sum(sleep_quality_components) / len(sleep_quality_components)

        # Work-life balance
        work_life_components = []
        if 'study_or_work_hours' in df_eng.columns:
            work_life_components.append(df_eng['study_or_work_hours'])
        if 'daily_commute_time_mins' in df_eng.columns:
            work_life_components.append(df_eng['daily_commute_time_mins']/60)

        if work_life_components:
            df_eng['work_life_balance'] = sum(work_life_components)

        # Ensure all features are numeric
        for col in df_eng.columns:
            if df_eng[col].dtype not in ['int8', 'int16', 'int32', 'int64', 'float16', 'float32', 'float64', 'bool']:
                df_eng[col] = df_eng[col].astype('float64')

        return df_eng
        
    except Exception as e:
        print(f"Error in feature engineering: {str(e)}")
        return df

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'model_loaded': trained_model is not None
    })

@app.route('/predict', methods=['POST'])
def predict():
    """Main prediction endpoint"""
    try:
        # Check if model is loaded
        if trained_model is None:
            print("ERROR: Model not loaded")
            return jsonify({
                'error': 'Model not loaded. Please ensure the model file exists.',
                'success': False
            }), 500

        # Get input data from request
        input_data = request.json
        print(f"DEBUG: Received input data: {input_data}")
        
        if not input_data:
            print("ERROR: No input data provided")
            return jsonify({
                'error': 'No input data provided',
                'success': False
            }), 400

        # Preprocess input
        print("DEBUG: Starting preprocessing...")
        processed_data = preprocess_input(input_data)
        if processed_data is None:
            print("ERROR: Preprocessing failed")
            return jsonify({
                'error': 'Error in data preprocessing',
                'success': False
            }), 400
        
        print(f"DEBUG: Processed data shape: {processed_data.shape}")
        print(f"DEBUG: Processed data columns: {list(processed_data.columns)}")
        print(f"DEBUG: Processed data sample: {processed_data.iloc[0].to_dict()}")

        # Apply feature engineering
        print("DEBUG: Starting feature engineering...")
        engineered_data = create_engineered_features(processed_data)
        print(f"DEBUG: Engineered data shape: {engineered_data.shape}")
        print(f"DEBUG: Engineered data columns: {list(engineered_data.columns)}")

        # Check expected vs actual features
        print(f"DEBUG: Expected features: {len(feature_columns)}")
        print(f"DEBUG: Expected feature names: {feature_columns[:10]}...")  # Show first 10
        
        # Ensure all required columns are present
        missing_cols = set(feature_columns) - set(engineered_data.columns)
        if missing_cols:
            print(f"DEBUG: Missing columns: {missing_cols}")
            for col in missing_cols:
                engineered_data[col] = 0

        # Reorder columns to match training data
        engineered_data = engineered_data[feature_columns]
        print(f"DEBUG: Final data shape for prediction: {engineered_data.shape}")
        
        # Show some sample values
        sample_values = engineered_data.iloc[0]
        non_zero_values = sample_values[sample_values != 0]
        print(f"DEBUG: Non-zero values in input: {dict(non_zero_values.head(10))}")

        # Make prediction
        print("DEBUG: Making prediction...")
        prediction = trained_model.predict(engineered_data)[0]
        prediction_proba = trained_model.predict_proba(engineered_data)[0]
        
        print(f"DEBUG: Raw prediction: {prediction}")
        print(f"DEBUG: Prediction probabilities: {prediction_proba}")

        # Determine risk level
        risk_probability = prediction_proba[1]
        if risk_probability < 0.3:
            risk_level = "Low"
            risk_description = "Low sleep deprivation risk detected"
        elif risk_probability < 0.7:
            risk_level = "Moderate" 
            risk_description = "Moderate sleep deprivation risk detected"
        else:
            risk_level = "High"
            risk_description = "High sleep deprivation risk detected"

        # Prepare response
        response = {
            'success': True,
            'prediction': int(prediction),
            'prediction_text': 'Sleep Deprivation Risk Detected' if prediction == 1 else 'No Sleep Deprivation Risk',
            'risk_level': risk_level,
            'risk_description': risk_description,
            'confidence': {
                'no_risk': float(prediction_proba[0]),
                'risk': float(prediction_proba[1])
            },
            'risk_percentage': float(risk_probability * 100)
        }
        
        print(f"DEBUG: Final response: {response}")
        return jsonify(response)

    except Exception as e:
        print(f"PREDICTION ERROR: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({
            'error': f'Prediction failed: {str(e)}',
            'success': False
        }), 500

if __name__ == '__main__':
    print("Loading model...")
    if load_model_and_preprocessors():
        print("Model loaded successfully!")
        print("Starting Flask server...")
        app.run(debug=True, host='0.0.0.0', port=5000)
    else:
        print("Failed to load model. Please ensure 'sleep_deprivation_model.pkl' exists.")
        print("Run your XGBoost training script first to create the model file.")