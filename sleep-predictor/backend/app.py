import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel
from typing import Optional
import joblib
from fastapi.middleware.cors import CORSMiddleware
import numpy as np

# -------- Load trained pipeline --------
pipeline = joblib.load("sleep_deprivation_pipeline.joblib")

# -------- FastAPI app --------
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000","http://127.0.0.1:3000"],  
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
def preprocess_user_input(user_dict):
    df_user = pd.DataFrame([user_dict])

    # Add engineered sleep duration
    df_user["sleep_duration_hours"] = df_user.apply(
        lambda row: compute_sleep_duration(row["bedtime"], row["wakeup_time"]), axis=1
    )

    # Convert numeric fields to floats/ints if not None
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
    print("✅ Incoming request:", data.dict())
    try:
        input_dict = data.dict()
        print("🚀 Received data in backend:", input_dict, flush=True)  # <-- debug print

        # Preprocess input
        processed_df = preprocess_user_input(input_dict)
        print("🛠 Processed DataFrame:", processed_df.head(), flush=True)  # <-- debug print

        # Make prediction
        prediction = pipeline.predict(processed_df)[0]
        proba = pipeline.predict_proba(processed_df)[0]

        result = {
            "prediction": int(prediction),
            "predictionText": "High Risk" if prediction == 1 else "No Risk",
            "riskLevel": "high" if prediction == 1 else "low",
            "riskDescription": "Signs of possible sleep deprivation" if prediction == 1 else "No major risks detected",
            "confidence": {
                "no_risk": float(proba[0]),
                "risk": float(proba[1])
            },
            "riskPercentage": float(proba[1] * 100)
        }
        print("✅ Result to send:", result, flush=True)  # <-- debug print
        return result

    except Exception as e:
        print("❌ Error occurred:", str(e), flush=True)  # <-- debug print
        return {"error": str(e)}

