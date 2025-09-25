import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel
from typing import Optional
import joblib
from fastapi.middleware.cors import CORSMiddleware
import numpy as np

# Import your ML helper
from ml_pipeline import predict_sleep_deprivation, preprocess_user_input, cat_cols, num_cols, X_train

# -------- Load trained pipeline --------
pipeline = joblib.load("sleep_deprivation_pipeline.joblib")

# -------- FastAPI app --------
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -------- Pydantic user input model --------
class UserInput(BaseModel):
    age: int
    gender: str
    bedtime: str
    wakeup_time: str
    sleep_latency: int
    screen_time_hours: Optional[float] = None
    caffeine_intake: Optional[float] = None
    physical_activity_mins: Optional[int] = None
    water_intake_liters: Optional[float] = None
    stress_level: Optional[int] = None
    sleep_consistency_score: Optional[int] = None
    daily_commute_time_mins: Optional[int] = None
    afternoon_naps: Optional[int] = None
    time_spent_outdoors_daily: Optional[int] = None
    social_media_hours: Optional[float] = None
    study_or_work_hours: Optional[float] = None
    diet_meal_timing: Optional[int] = None
    in_bed_phone_use_percent: Optional[int] = None
    alcohol_habit: Optional[str] = None
    smoking: Optional[str] = None
    occupation_type: Optional[str] = None
    work_shift_type: Optional[str] = None
    preferred_sleep_time_category: Optional[str] = None
    day_type: Optional[str] = None
    light_exposure_before_bed: Optional[str] = None
    medical_conditions: Optional[str] = None

# -------- Helper: compute sleep duration --------
def compute_sleep_duration(bed, wake):
    try:
        bh, bm = map(int, str(bed).split(":"))
        wh, wm = map(int, str(wake).split(":"))
        bed_minutes = bh * 60 + bm
        wake_minutes = wh * 60 + wm
        return ((wake_minutes - bed_minutes) % (24 * 60)) / 60.0
    except:
        return np.nan

# -------- Preprocessing function for user input --------
def preprocess_user_input_backend(user_dict):
    df_user = pd.DataFrame([user_dict])
    # Add engineered sleep duration
    df_user["sleep_duration_hours"] = df_user.apply(
        lambda row: compute_sleep_duration(row["bedtime"], row["wakeup_time"]), axis=1
    )
    # Fill numeric fields
    numeric_fields = [
        "age", "sleep_latency", "screen_time_hours", "in_bed_phone_use_percent",
        "caffeine_intake", "physical_activity_mins", "diet_meal_timing",
        "water_intake_liters", "stress_level", "study_or_work_hours",
        "social_media_hours", "sleep_consistency_score",
        "daily_commute_time_mins", "afternoon_naps", "time_spent_outdoors_daily"
    ]
    for field in numeric_fields:
        if field in df_user.columns:
            df_user[field] = pd.to_numeric(df_user[field], errors='coerce').fillna(0)
    # Drop raw time columns
    df_user = df_user.drop(columns=["bedtime", "wakeup_time"], errors="ignore")
    return df_user

# -------- Prediction endpoint --------
@app.post("/predict")
def predict(data: UserInput):
    try:
        input_dict = data.dict()
        print("✅ Incoming request:", input_dict, flush=True)

        # Call your helper to get prediction + SHAP top factors
        shap_result = predict_sleep_deprivation(
            user_input=input_dict,
            pipeline=pipeline,
            X_reference=X_train,  # reference data needed for SHAP
            cat_cols=cat_cols,
            num_cols=num_cols,
            verbose=False,
            top_n_features=5
        )

        # Build final response
        prediction = 1 if shap_result["prediction"] == "Sleep Deprived" else 0
        prob = shap_result.get("probability", 0.0)

        result = {
            "prediction": prediction,
            "predictionText": shap_result["prediction"],
            "riskLevel": "high" if prediction == 1 else "low",
            "riskDescription": "Signs of possible sleep deprivation" if prediction == 1 else "No major risks detected",
            "confidence": {
                "no_risk": round(1 - prob, 3),
                "risk": round(prob, 3)
            },
            "riskPercentage": round(prob * 100, 2),
            "top_factors": shap_result.get("top_factors", [])
        }

        print("✅ Result to send:", result, flush=True)
        return result

    except Exception as e:
        print("❌ Error occurred:", str(e), flush=True)
        return {"error": str(e)}
