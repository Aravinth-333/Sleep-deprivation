# -------- Full corrected pipeline with leakage removed --------
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import lightgbm as lgb
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

# ---------- Load data ----------
file_path = "C:/Users/aravi/OneDrive/Desktop/Sleep/sleep-predictor/backend/final_dataset1.csv"
df = pd.read_csv(file_path)

# ---------- Feature Engineering ----------
def compute_sleep_duration(bed, wake):
    try:
        bh, bm = map(int, str(bed).split(":"))
        wh, wm = map(int, str(wake).split(":"))
        bed_minutes = bh * 60 + bm
        wake_minutes = wh * 60 + wm
        return ((wake_minutes - bed_minutes) % (24 * 60)) / 60.0
    except:
        return np.nan

df["sleep_duration_hours"] = df.apply(
    lambda row: compute_sleep_duration(row["bedtime"], row["wakeup_time"]), axis=1
)

# Drop raw string cols
df = df.drop(columns=["bedtime", "wakeup_time"], errors="ignore")

# ---------- Prepare X, y ----------
target = "sleep_deprivation"
if target not in df.columns:
    raise ValueError(f"Target column '{target}' not found in dataframe.")

df.dropna(subset=[target], inplace=True)

X = df.drop(columns=[target]).copy()
y = df[target].copy()

# ---------- Explicit Feature Groups (NO leakage features) ----------
cat_cols = [
    "gender", "day_type", "occupation_type",
    "preferred_sleep_time_category", "light_exposure_before_bed",
    "smoking", "alcohol_habit", "medical_conditions", "work_shift_type"
]

num_cols = [
    "age", "sleep_latency", "screen_time_hours", "in_bed_phone_use_percent",
    "caffeine_intake", "physical_activity_mins", "diet_meal_timing",
    "water_intake_liters", "stress_level", "study_or_work_hours",
    "social_media_hours", "sleep_consistency_score",
    "daily_commute_time_mins", "afternoon_naps", "time_spent_outdoors_daily"
]

# 🚨 Explicitly drop leakage features
leakage_features = ["sleep_quality", "energy_level", "sleep_duration_hours"]
X = X.drop(columns=leakage_features, errors="ignore")
num_cols = [c for c in num_cols if c not in leakage_features]

# ---------- Normalize study_or_work_hours ----------
X["study_or_work_hours"] = X["study_or_work_hours"].astype(float).round(1)

# ---------- Preprocessor & Pipeline ----------
preprocessor = ColumnTransformer(
    transformers=[
        ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
        ("num", StandardScaler(), num_cols),
    ],
    remainder="drop"
)

pipeline = Pipeline([
    ("preprocessor", preprocessor),
    ("classifier", lgb.LGBMClassifier(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=5,
        num_leaves=20,
        min_child_samples=50,
        subsample=0.7,
        colsample_bytree=0.7,
        reg_alpha=0.5,
        reg_lambda=0.5,
        random_state=42
    ))
])

# ---------- Train/Test split ----------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)

# ---------- Fit model ----------
pipeline.fit(X_train, y_train)

# ---------- Evaluate ----------
y_pred = pipeline.predict(X_test)
acc = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
cm = confusion_matrix(y_test, y_pred)

print("Accuracy:", acc)
print("F1 Score:", f1)
print("Confusion Matrix:\n", cm)
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# ---------- Confusion Matrix Plot ----------
plt.figure(figsize=(6,4))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=["No Deprivation","Deprivation"],
            yticklabels=["No Deprivation","Deprivation"])
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.show()


# ---------- Feature Importances ----------
ohe = pipeline.named_steps['preprocessor'].named_transformers_['cat']
try:
    ohe_names = ohe.get_feature_names_out(cat_cols).tolist()
except Exception:
    ohe_names = []
    for i, c in enumerate(cat_cols):
        vals = X_train[c].astype(str).unique()[:20]
        ohe_names += [f"{c}__{v}" for v in vals]

feature_names = ohe_names + num_cols

clf = pipeline.named_steps['classifier']
importances = clf.feature_importances_
if len(importances) == len(feature_names):
    fi_series = pd.Series(importances, index=feature_names).sort_values(ascending=False)
    plt.figure(figsize=(10,6))
    fi_series.head(15).plot(kind="bar")
    plt.title("Top 15 Feature Importances")
    plt.ylabel("Importance Score")
    plt.xlabel("Features")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.show()
else:
    print("Warning: number of importances != number of feature names. Skipping importance plot.")

# ---------- Save pipeline ----------
joblib.dump(pipeline, "sleep_deprivation_pipeline.joblib")
print("Saved pipeline to sleep_deprivation_pipeline.joblib")

# ---------- User Preprocessing Function ----------
def preprocess_user_input(user_dict):
    df_user = pd.DataFrame([user_dict])

    # Add engineered sleep duration
    df_user["sleep_duration_hours"] = df_user.apply(
        lambda row: compute_sleep_duration(row["bedtime"], row["wakeup_time"]), axis=1
    )

    # Normalize study_or_work_hours
    df_user["study_or_work_hours"] = df_user["study_or_work_hours"].astype(float).round(1)

    # Drop raw cols
    df_user = df_user.drop(columns=["bedtime", "wakeup_time"], errors="ignore")
    return df_user
import re
import pandas as pd
import shap
import numpy as np


def preprocess_user_input(user_dict):
    df_user = pd.DataFrame([user_dict])

    # Add engineered sleep duration
    df_user["sleep_duration_hours"] = df_user.apply(
        lambda row: compute_sleep_duration(row["bedtime"], row["wakeup_time"]), axis=1
    )

    # Normalize study_or_work_hours
    df_user["study_or_work_hours"] = df_user["study_or_work_hours"].astype(float).round(1)

    # Drop raw bedtime/wakeup_time
    df_user = df_user.drop(columns=["bedtime", "wakeup_time"], errors="ignore")
    return df_user.to_dict(orient="records")[0]

def _normalize_str(s):
    """Lowercase and remove non-alphanumeric so comparisons are robust."""
    return re.sub(r'\W+', '', str(s).lower())

def predict_sleep_deprivation(user_input: dict, pipeline, X_reference, cat_cols, num_cols,
                              fill_cat="Unknown", fill_num=0, verbose=True, top_n_features=5):
    # 0. normalize user_input keys -> keep as-is for display, but have normalized keys list
    user_keys = list(user_input.keys())
    user_keys_sorted = sorted(user_keys, key=len, reverse=True)  # prefer longest match

    # 1. Prepare input DataFrame (same as before)
    custom_df = pd.DataFrame([user_input]).astype(object)
    missing = []
    for col in X_reference.columns:
        if col not in custom_df.columns:
            if col in cat_cols:
                custom_df[col] = fill_cat
                missing.append((col, fill_cat))
            else:
                custom_df[col] = fill_num
                missing.append((col, fill_num))

    for c in num_cols:
        custom_df[c] = pd.to_numeric(custom_df[c], errors='coerce').fillna(fill_num)
    for c in cat_cols:
        custom_df[c] = custom_df[c].astype(str)
    custom_df = custom_df[X_reference.columns]

    if verbose and missing:
        print("Auto-filled missing fields:", missing)

    # 2. Predict
    pred = pipeline.predict(custom_df)[0]
    prob = pipeline.predict_proba(custom_df)[0][1] if hasattr(pipeline, "predict_proba") else None

    # 3. SHAP explanation (robust mapping)
    try:
        preprocessor = pipeline.named_steps['preprocessor']
        model = pipeline.named_steps['classifier']

        # transform and ensure dense
        X_transformed = preprocessor.transform(custom_df)
        if hasattr(X_transformed, "toarray"):
            X_transformed = X_transformed.toarray()

        # build feature names (numeric + ohe names)
        feature_names = []
        for name, transformer, columns in preprocessor.transformers_:
            if name == 'num':
                feature_names.extend(columns)
            elif name == 'cat':
                ohe = transformer
                # get_feature_names_out returns names like "col_category" or "col category" depending on sklearn
                feature_names.extend(ohe.get_feature_names_out(columns))

        # SHAP on the classifier with transformed numeric data
        explainer = shap.TreeExplainer(model)
        shap_values = explainer(X_transformed)

        # pick top contributions
        contributions = sorted(
            zip(feature_names, shap_values.values[0]),
            key=lambda x: abs(x[1]),
            reverse=True
        )[:top_n_features]

        # Build user-friendly list — only show categorical OHE entries that match the user's actual input
        top_factors = []
        for feat, val in contributions:
            # 1) If exact feature name equals a user key -> numeric feature
            if feat in user_input:
                display_value = f"{feat} = {user_input.get(feat)}"
                top_factors.append(
                    f"{display_value} → {'higher' if val > 0 else 'lower'} risk (impact: {abs(val):.3f})"
                )
                continue

            # 2) Try to match this OHE feature to one of the user's categorical keys
            matched = False
            for key in user_keys_sorted:
                # try common separators used by get_feature_names_out
                for sep in ("__", "_", "=", " "):
                    prefix = f"{key}{sep}"
                    if feat.startswith(prefix):
                        base = key
                        category = feat[len(prefix):]
                        # normalize both to compare robustly
                        if _normalize_str(category) == _normalize_str(user_input.get(base, "")):
                            display_value = f"{base} = {user_input.get(base)}"
                            top_factors.append(
                                f"{display_value} → {'higher' if val > 0 else 'lower'} risk (impact: {abs(val):.3f})"
                            )
                            matched = True
                        # whether matched or not, we don't want to try smaller keys if this was the prefix match
                        break
                if matched:
                    break

            # 3) If not matched and feat looks like an OHE column but user didn't choose it, skip it.
            # If not matched and feat isn't OHE style, optionally include numeric fallback if column name without suffix is present:
            if not matched:
                # fallback: if feat contains one of the user keys anywhere, try to extract base
                for key in user_keys_sorted:
                    if key in feat:
                        # e.g., feat = "medical_conditions_Sleep Apnea" -> key "medical_conditions"
                        # if user's value equals extracted category, we would have matched above; so skip.
                        pass
                # do nothing (skip irrelevant OHE columns)

        # final safety: if no top_factors collected, show top numeric contributions (best-effort)
        if not top_factors:
            for feat, val in contributions:
                # try to show numeric if present in user_input (last resort)
                if feat in user_input:
                    top_factors.append(
                        f"{feat} = {user_input.get(feat)} → {'higher' if val > 0 else 'lower'} risk (impact: {abs(val):.3f})"
                    )
            if not top_factors:
                top_factors = ["No matching SHAP contributions for the exact user inputs."]

    except Exception as e:
        top_factors = ["SHAP explanation not available: " + str(e)]

    return {
        "prediction": "Sleep Deprived" if pred == 1 else "Not Sleep Deprived",
        "probability": round(float(prob), 3) if prob is not None else None,
        "top_factors": top_factors
    }


# ---------- Example usage (assuming pipeline, X_train, cat_cols, num_cols are defined from previous cells) ----------
high_sleep_deprivation_user = {
    'age': 30,
    'gender': 'Female',
    'bedtime': '23:45',
    'wakeup_time': '05:45',
    'sleep_latency': 25,
    'screen_time_hours': 3,
    'in_bed_phone_use_percent': 35,
    'caffeine_intake': 120,
    'physical_activity_mins': 45,
    'diet_meal_timing': 3,
    'water_intake_liters': 2.0,
    'stress_level': 6,
    'day_type': 'Weekday',
    'occupation_type': 'Technology Professional',
    'study_or_work_hours': 8,
    'energy_level': 5,
    'social_media_hours': 2,
    'light_exposure_before_bed': 'Medium',
    'sleep_consistency_score': 6,
    'daily_commute_time_mins': 35,
    'afternoon_naps': 20,
    'time_spent_outdoors_daily': 45,
    'smoking': 'Never',
    'alcohol_habit': 'Occasionally',
    'medical_conditions': 'None',
    'work_shift_type': 'Remote Work',
    'preferred_sleep_time_category': 'Normal Sleeper',
    'sleep_quality': 5
}

user_processed = preprocess_user_input(high_sleep_deprivation_user)

result = predict_sleep_deprivation(
    user_processed,
    pipeline,
    X_train,
    cat_cols,
    num_cols,
    verbose=True
)


print("Prediction result:", result)