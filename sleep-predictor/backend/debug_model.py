# debug_model.py - Check what features your model expects
import joblib
import pandas as pd

def debug_model():
    try:
        # Load the model
        print("Loading model...")
        model_data = joblib.load('sleep_deprivation_model.pkl')
        
        model = model_data['model']
        feature_columns = model_data['feature_columns']
        label_encoders = model_data.get('label_encoders', {})
        
        print(f"✅ Model loaded successfully!")
        print(f"📊 Total expected features: {len(feature_columns)}")
        print("\n🏷️ Expected feature names:")
        for i, feature in enumerate(feature_columns, 1):
            print(f"{i:2d}. {feature}")
        
        print(f"\n🔧 Label encoders available:")
        for col, encoder in label_encoders.items():
            print(f"  {col}: {list(encoder.classes_)}")
        
        # Test with sample data that should trigger high risk
        print("\n🧪 Testing with high-risk profile...")
        
        test_data = {
  "age": 25,
  "gender": "Male",
  "sleep_hours": 4.5,
  "sleep_quality": 3,
  "bedtime": "02:00",
  "wakeup_time": "06:30",
  "sleep_latency": 60,
  "screen_time_hours": 12,
  "in_bed_phone_use_percent": 90,
  "caffeine_intake": 400,
  "physical_activity_mins": 10,
  "diet_meal_timing": 1,
  "water_intake_liters": 1.0,
  "stress_level": 9,
  "day_type": "Weekday",
  "occupation_type": "Healthcare Professional",
  "study_or_work_hours": 12,
  "energy_level": 2,
  "preferred_sleep_time_category": "Night Owl",
  "social_media_hours": 6,
  "light_exposure_before_bed": "High",
  "sleep_consistency_score": 2,
  "daily_commute_time_mins": 90,
  "afternoon_naps": 0,
  "time_spent_outdoors_daily": 15,
  "smoking": "Regular",
  "alcohol_habit": "Heavy",
  "sleep_environment_quality": "Poor",
  "medical_conditions": "Anxiety",
  "work_shift_type": "Night Shift"
        }
        
        # Test preprocessing
        df = pd.DataFrame([test_data])
        print(f"Original data columns: {list(df.columns)}")
        
        # Apply same preprocessing as training
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
            print(f"Bedtime converted: {test_data['bedtime']} -> {df['bedtime_minutes'].iloc[0]} minutes")

        if 'wakeup_time' in df.columns:
            df['wakeup_minutes'] = df['wakeup_time'].apply(time_to_minutes)
            df = df.drop('wakeup_time', axis=1)
            print(f"Wakeup time converted: {test_data['wakeup_time']} -> {df['wakeup_minutes'].iloc[0]} minutes")
        
        print(f"After time conversion: {list(df.columns)}")
        
        # Apply label encoders
        for col in df.columns:
            if col in label_encoders and df[col].dtype == 'object':
                le = label_encoders[col]
                original_value = df[col].iloc[0]
                try:
                    df[col] = le.transform(df[col].astype(str))
                    print(f"✅ Encoded {col}: '{original_value}' -> {df[col].iloc[0]}")
                except ValueError as e:
                    print(f"❌ Failed to encode {col}: '{original_value}' - {str(e)}")
                    df[col] = 0
        
        # Convert to numeric
        for col in df.columns:
            if df[col].dtype == 'object':
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        df = df.fillna(0)
        
        # Feature engineering (simplified)
        if 'sleep_hours' in df.columns and 'sleep_latency' in df.columns:
            df['sleep_efficiency'] = df['sleep_hours'] / (df['sleep_hours'] + df['sleep_latency']/60 + 1e-6)
            print(f"Sleep efficiency: {df['sleep_efficiency'].iloc[0]:.3f}")
        
        if 'screen_time_hours' in df.columns and 'social_media_hours' in df.columns:
            df['total_screen_time'] = df['screen_time_hours'] + df['social_media_hours']
            print(f"Total screen time: {df['total_screen_time'].iloc[0]} hours")
        
        print(f"After feature engineering: {list(df.columns)}")
        
        # Add missing columns
        missing_cols = set(feature_columns) - set(df.columns)
        if missing_cols:
            print(f"⚠️ Adding missing columns: {missing_cols}")
            for col in missing_cols:
                df[col] = 0
        
        # Reorder columns
        df = df[feature_columns]
        
        # Check final values
        sample_row = df.iloc[0]
        non_zero_features = sample_row[sample_row != 0]
        print(f"\n📊 Non-zero features ({len(non_zero_features)}):")
        for feature, value in non_zero_features.items():
            print(f"  {feature}: {value}")
        
        # Make prediction
        prediction = model.predict(df)[0]
        prediction_proba = model.predict_proba(df)[0]
        
        print(f"\n🎯 PREDICTION RESULTS:")
        print(f"Raw prediction: {prediction}")
        print(f"Probabilities: [No Risk: {prediction_proba[0]:.3f}, Risk: {prediction_proba[1]:.3f}]")
        print(f"Risk percentage: {prediction_proba[1]*100:.1f}%")
        
        if prediction_proba[1] < 0.3:
            risk_level = "Low"
        elif prediction_proba[1] < 0.7:
            risk_level = "Moderate"
        else:
            risk_level = "High"
        
        print(f"Risk level: {risk_level}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("🔍 DEBUGGING MODEL FEATURES AND PREDICTION")
    print("=" * 60)
    debug_model()