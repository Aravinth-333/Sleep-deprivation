import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import (classification_report, confusion_matrix,
                           roc_auc_score, roc_curve, precision_recall_curve,
                           accuracy_score, precision_score, recall_score, f1_score)
import xgboost as xgb
from xgboost import XGBClassifier
import warnings
warnings.filterwarnings('ignore')

# Set style for plots
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

# Global variables to store model and preprocessing info
trained_model = None
feature_columns = None
label_encoders = None

def load_and_preprocess_data(file_path='organized_sleep_dataset.csv'):
    """
    Load and preprocess the sleep dataset for XGBoost training
    """
    print("🔄 Loading and preprocessing data...")

    # Load data
    df = pd.read_csv(file_path)
    print(f"📊 Dataset shape: {df.shape}")
    print(f"🎯 Sleep deprivation cases: {df['sleep_deprivation'].sum()} ({df['sleep_deprivation'].mean()*100:.1f}%)")

    # Print column info for debugging
    print(f"📋 Columns in dataset: {list(df.columns)}")
    print(f"📋 Data types:\n{df.dtypes}")

    # Separate features and target
    target = df['sleep_deprivation']
    features = df.drop('sleep_deprivation', axis=1)

    # Handle time features (bedtime and wakeup_time) if they exist
    if 'bedtime' in features.columns:
        def time_to_minutes(time_str):
            """Convert time string to minutes from midnight"""
            try:
                if pd.isna(time_str):
                    return 0
                # Handle different time formats
                time_str = str(time_str).strip()
                if ':' in time_str:
                    hours, minutes = map(int, time_str.split(':'))
                    return hours * 60 + minutes
                else:
                    # Assume it's already in numeric format
                    return float(time_str)
            except:
                return 0

        features['bedtime_minutes'] = features['bedtime'].apply(time_to_minutes)
        features = features.drop('bedtime', axis=1)

    if 'wakeup_time' in features.columns:
        features['wakeup_minutes'] = features['wakeup_time'].apply(time_to_minutes)
        features = features.drop('wakeup_time', axis=1)

    # Calculate sleep duration consistency if user_id exists
    if 'user_id' in features.columns and 'sleep_hours' in features.columns:
        user_avg_sleep = features.groupby('user_id')['sleep_hours'].mean()
        features['sleep_deviation'] = features.apply(
            lambda row: abs(row['sleep_hours'] - user_avg_sleep.get(row['user_id'], row['sleep_hours'])), axis=1
        )

    # Handle date features if they exist
    if 'record_date' in features.columns:
        features['record_date'] = pd.to_datetime(features['record_date'], errors='coerce')
        features['day_of_week'] = features['record_date'].dt.dayofweek
        features['month'] = features['record_date'].dt.month
        features = features.drop('record_date', axis=1)

    # Identify all object/categorical columns
    categorical_columns = []
    for col in features.columns:
        if features[col].dtype == 'object' or features[col].dtype.name == 'category':
            categorical_columns.append(col)

    print(f"🔍 Found categorical columns: {categorical_columns}")

    # Encode all categorical variables
    label_encoders = {}
    for col in categorical_columns:
        print(f"🔄 Encoding column: {col}")
        le = LabelEncoder()
        # Handle NaN values by converting to string first
        features[col] = features[col].fillna('unknown')
        features[col] = le.fit_transform(features[col].astype(str))
        label_encoders[col] = le

    # Convert any remaining object columns to numeric
    for col in features.columns:
        if features[col].dtype == 'object':
            print(f"⚠️  Force converting object column {col} to numeric")
            try:
                # Try direct numeric conversion first
                features[col] = pd.to_numeric(features[col], errors='coerce')
            except:
                # If that fails, use label encoding
                le = LabelEncoder()
                features[col] = features[col].fillna('unknown')
                features[col] = le.fit_transform(features[col].astype(str))
                label_encoders[col] = le

    # Handle any NaN values that might have been created
    features = features.fillna(0)

    # Ensure all columns are numeric types that XGBoost can handle
    for col in features.columns:
        if features[col].dtype not in ['int8', 'int16', 'int32', 'int64', 'float16', 'float32', 'float64', 'bool']:
            print(f"🔧 Converting {col} from {features[col].dtype} to float64")
            features[col] = features[col].astype('float64')

    print(f"✅ Preprocessing complete. Features shape: {features.shape}")
    print(f"📊 Final data types: {features.dtypes.value_counts().to_dict()}")

    return features, target, label_encoders

def create_feature_engineering(X):
    """
    Create additional engineered features
    """
    print("🔧 Creating engineered features...")

    X_eng = X.copy()

    # Sleep efficiency features (if sleep_hours and sleep_latency exist)
    if 'sleep_hours' in X_eng.columns and 'sleep_latency' in X_eng.columns:
        X_eng['sleep_efficiency'] = X_eng['sleep_hours'] / (X_eng['sleep_hours'] + X_eng['sleep_latency']/60 + 1e-6)

    # Technology usage intensity
    if 'screen_time_hours' in X_eng.columns and 'social_media_hours' in X_eng.columns:
        X_eng['total_screen_time'] = X_eng['screen_time_hours'] + X_eng['social_media_hours']

    if 'in_bed_phone_use_percent' in X_eng.columns and 'total_screen_time' in X_eng.columns:
        X_eng['digital_dependency'] = (X_eng['in_bed_phone_use_percent'] / 100) * X_eng['total_screen_time']

    # Health & lifestyle score
    health_components = []
    if 'physical_activity_mins' in X_eng.columns:
        health_components.append(X_eng['physical_activity_mins'] / 60)
    if 'water_intake_liters' in X_eng.columns:
        health_components.append(X_eng['water_intake_liters'] / 3)
    if 'time_spent_outdoors_daily' in X_eng.columns:
        health_components.append(X_eng['time_spent_outdoors_daily'] / 120)
    if 'stress_level' in X_eng.columns:
        health_components.append(-X_eng['stress_level'] / 10)
    if 'caffeine_intake' in X_eng.columns:
        health_components.append(-X_eng['caffeine_intake'] / 400)

    if health_components:
        X_eng['health_score'] = sum(health_components)

    # Sleep quality composite
    sleep_quality_components = []
    if 'sleep_quality' in X_eng.columns:
        sleep_quality_components.append(X_eng['sleep_quality'])
    if 'sleep_consistency_score' in X_eng.columns:
        sleep_quality_components.append(X_eng['sleep_consistency_score'])
    if 'sleep_latency' in X_eng.columns:
        sleep_quality_components.append(-X_eng['sleep_latency'] / 30)

    if sleep_quality_components:
        X_eng['sleep_quality_composite'] = sum(sleep_quality_components) / len(sleep_quality_components)

    # Work-life balance indicator
    work_life_components = []
    if 'study_or_work_hours' in X_eng.columns:
        work_life_components.append(X_eng['study_or_work_hours'])
    if 'daily_commute_time_mins' in X_eng.columns:
        work_life_components.append(X_eng['daily_commute_time_mins']/60)

    if work_life_components:
        X_eng['work_life_balance'] = sum(work_life_components)

    # Ensure all new features are numeric
    for col in X_eng.columns:
        if X_eng[col].dtype not in ['int8', 'int16', 'int32', 'int64', 'float16', 'float32', 'float64', 'bool']:
            X_eng[col] = X_eng[col].astype('float64')

    print(f"✅ Feature engineering complete. New shape: {X_eng.shape}")
    return X_eng

def train_xgboost_model(X_train, X_test, y_train, y_test, hyperparameter_tuning=True):
    """
    Train XGBoost model with optional hyperparameter tuning
    """
    print("🚀 Training XGBoost model...")

    # Verify data types before training
    print("🔍 Verifying data types before training...")
    for col in X_train.columns:
        if X_train[col].dtype == 'object':
            print(f"❌ Column {col} is still object type: {X_train[col].dtype}")
            return None

    print(f"✅ All columns are numeric: {X_train.dtypes.value_counts().to_dict()}")

    if hyperparameter_tuning:
        print("🔍 Performing hyperparameter tuning...")

        # Define parameter grid
        param_grid = {
            'n_estimators': [100, 200, 300],
            'max_depth': [3, 6, 9],
            'learning_rate': [0.01, 0.1, 0.2],
            'subsample': [0.8, 0.9, 1.0],
            'colsample_bytree': [0.8, 0.9, 1.0]
        }

        # Initialize XGBoost
        xgb_model = XGBClassifier(
            objective='binary:logistic',
            random_state=42,
            eval_metric='logloss',
            enable_categorical=False  # Ensure we're not using categorical features
        )

        # Grid search with cross-validation
        grid_search = GridSearchCV(
            estimator=xgb_model,
            param_grid=param_grid,
            scoring='roc_auc',
            cv=5,
            n_jobs=-1,
            verbose=1
        )

        grid_search.fit(X_train, y_train)
        best_model = grid_search.best_estimator_

        print(f"🎯 Best parameters: {grid_search.best_params_}")
        print(f"🎯 Best CV score: {grid_search.best_score_:.4f}")

    else:
        # Use default parameters with some optimization
        best_model = XGBClassifier(
            n_estimators=200,
            max_depth=6,
            learning_rate=0.1,
            subsample=0.9,
            colsample_bytree=0.9,
            objective='binary:logistic',
            random_state=42,
            eval_metric='logloss',
            enable_categorical=False  # Explicitly disable categorical features
        )

        # Train the model
        print("🔄 Training model...")
        try:
            # For newer XGBoost versions
            best_model.fit(
                X_train, y_train,
                eval_set=[(X_test, y_test)],
                verbose=False
            )
        except Exception as e:
            # Fallback for older versions or other issues
            print(f"⚠️  Using fallback training method: {str(e)[:100]}...")
            best_model.fit(X_train, y_train)

    print("✅ Model training complete!")
    return best_model

def evaluate_model(model, X_test, y_test, X_train, y_train):
    """
    Comprehensive model evaluation
    """
    print("📊 Evaluating model performance...")

    # Predictions
    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)
    y_pred_proba_test = model.predict_proba(X_test)[:, 1]

    # Calculate metrics
    metrics = {
        'Train Accuracy': accuracy_score(y_train, y_pred_train),
        'Test Accuracy': accuracy_score(y_test, y_pred_test),
        'Precision': precision_score(y_test, y_pred_test),
        'Recall': recall_score(y_test, y_pred_test),
        'F1-Score': f1_score(y_test, y_pred_test),
        'ROC-AUC': roc_auc_score(y_test, y_pred_proba_test)
    }

    print("\n🎯 MODEL PERFORMANCE METRICS:")
    print("=" * 40)
    for metric, value in metrics.items():
        print(f"{metric:<15}: {value:.4f}")

    # Classification report
    print(f"\n📋 DETAILED CLASSIFICATION REPORT:")
    print("=" * 50)
    print(classification_report(y_test, y_pred_test))

    return metrics, y_pred_test, y_pred_proba_test

def plot_model_analysis(model, X_test, y_test, y_pred_proba, feature_names):
    """
    Create comprehensive model analysis plots
    """
    print("📈 Creating model analysis plots...")

    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('XGBoost Sleep Deprivation Model Analysis', fontsize=16, fontweight='bold')

    # 1. Confusion Matrix
    cm = confusion_matrix(y_test, (y_pred_proba > 0.5).astype(int))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[0,0])
    axes[0,0].set_title('Confusion Matrix')
    axes[0,0].set_xlabel('Predicted')
    axes[0,0].set_ylabel('Actual')

    # 2. ROC Curve
    fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
    roc_auc = roc_auc_score(y_test, y_pred_proba)
    axes[0,1].plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.3f})')
    axes[0,1].plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    axes[0,1].set_xlim([0.0, 1.0])
    axes[0,1].set_ylim([0.0, 1.05])
    axes[0,1].set_xlabel('False Positive Rate')
    axes[0,1].set_ylabel('True Positive Rate')
    axes[0,1].set_title('ROC Curve')
    axes[0,1].legend(loc="lower right")

    # 3. Precision-Recall Curve
    precision, recall, _ = precision_recall_curve(y_test, y_pred_proba)
    axes[0,2].plot(recall, precision, color='red', lw=2)
    axes[0,2].set_xlabel('Recall')
    axes[0,2].set_ylabel('Precision')
    axes[0,2].set_title('Precision-Recall Curve')

    # 4. Feature Importance (Top 15)
    feature_importance = model.feature_importances_
    feature_importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance': feature_importance
    }).sort_values('importance', ascending=True).tail(15)

    axes[1,0].barh(feature_importance_df['feature'], feature_importance_df['importance'])
    axes[1,0].set_title('Top 15 Feature Importance')
    axes[1,0].set_xlabel('Importance')

    # 5. Prediction Distribution
    axes[1,1].hist(y_pred_proba[y_test == 0], bins=30, alpha=0.7, label='No Sleep Deprivation', color='green')
    axes[1,1].hist(y_pred_proba[y_test == 1], bins=30, alpha=0.7, label='Sleep Deprivation', color='red')
    axes[1,1].set_xlabel('Predicted Probability')
    axes[1,1].set_ylabel('Frequency')
    axes[1,1].set_title('Prediction Distribution')
    axes[1,1].legend()

    # 6. Model Performance by Threshold
    thresholds = np.arange(0.1, 1.0, 0.05)
    precision_scores = []
    recall_scores = []
    f1_scores = []

    for threshold in thresholds:
        y_pred_thresh = (y_pred_proba > threshold).astype(int)
        precision_scores.append(precision_score(y_test, y_pred_thresh))
        recall_scores.append(recall_score(y_test, y_pred_thresh))
        f1_scores.append(f1_score(y_test, y_pred_thresh))

    axes[1,2].plot(thresholds, precision_scores, label='Precision', marker='o')
    axes[1,2].plot(thresholds, recall_scores, label='Recall', marker='s')
    axes[1,2].plot(thresholds, f1_scores, label='F1-Score', marker='^')
    axes[1,2].set_xlabel('Threshold')
    axes[1,2].set_ylabel('Score')
    axes[1,2].set_title('Performance vs Threshold')
    axes[1,2].legend()
    axes[1,2].grid(True)

    plt.tight_layout()
    plt.show()

    return feature_importance_df

def analyze_feature_importance(model, feature_names, label_encoders):
    """
    Detailed analysis of feature importance and insights
    """
    print("\n🔍 FEATURE IMPORTANCE ANALYSIS:")
    print("=" * 50)

    # Get feature importance
    feature_importance = model.feature_importances_
    feature_importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance': feature_importance
    }).sort_values('importance', ascending=False)

    # Top 10 most important features
    print("\n🏆 TOP 10 MOST IMPORTANT FEATURES:")
    for i, (_, row) in enumerate(feature_importance_df.head(10).iterrows(), 1):
        print(f"{i:2d}. {row['feature']:<25} : {row['importance']:.4f}")

    # Group features by category and analyze
    feature_categories = {
        'Sleep Metrics': ['sleep_hours', 'sleep_quality', 'sleep_latency', 'sleep_consistency_score',
                         'sleep_efficiency', 'sleep_quality_composite', 'sleep_deviation'],
        'Digital Behavior': ['screen_time_hours', 'social_media_hours', 'in_bed_phone_use_percent',
                            'light_exposure_before_bed', 'total_screen_time', 'digital_dependency'],
        'Physical Health': ['physical_activity_mins', 'energy_level', 'time_spent_outdoors_daily',
                           'afternoon_naps', 'health_score'],
        'Lifestyle': ['stress_level', 'caffeine_intake', 'water_intake_liters', 'diet_meal_timing'],
        'Demographics': ['age', 'gender', 'occupation_type', 'work_shift_type', 'medical_conditions', 'age_group'],
        'Work/Schedule': ['study_or_work_hours', 'daily_commute_time_mins', 'day_type', 'work_life_balance']
    }

    print(f"\n📊 FEATURE IMPORTANCE BY CATEGORY:")
    category_importance = {}
    for category, features in feature_categories.items():
        category_features = [f for f in features if f in feature_names]
        if category_features:
            category_score = feature_importance_df[
                feature_importance_df['feature'].isin(category_features)
            ]['importance'].sum()
            category_importance[category] = category_score
            print(f"{category:<20}: {category_score:.4f}")

    return feature_importance_df, category_importance

def store_model_info(model, X_engineered, encoders):
    """
    Store model and preprocessing information for manual testing
    """
    global trained_model, feature_columns, label_encoders

    trained_model = model
    feature_columns = list(X_engineered.columns)
    label_encoders = encoders

    print(f"✅ Model and preprocessing info stored for manual testing.")
    print(f"📊 Feature columns saved: {len(feature_columns)}")

def save_model(model, filename='sleep_deprivation_model.pkl'):
    """
    Save the trained model to disk
    """
    try:
        import joblib
        joblib.dump({
            'model': model,
            'feature_columns': feature_columns,
            'label_encoders': label_encoders
        }, filename)
        print(f"💾 Model saved successfully as '{filename}'")
    except ImportError:
        print("⚠️  joblib not available. Model not saved to disk.")
    except Exception as e:
        print(f"❌ Error saving model: {str(e)}")

def load_model(filename='sleep_deprivation_model.pkl'):
    """
    Load a previously trained model
    """
    global trained_model, feature_columns, label_encoders

    try:
        import joblib
        model_data = joblib.load(filename)
        trained_model = model_data['model']
        feature_columns = model_data['feature_columns']
        label_encoders = model_data['label_encoders']
        print(f"✅ Model loaded successfully from '{filename}'")
        return trained_model
    except ImportError:
        print("❌ joblib not available. Cannot load model.")
        return None
    except Exception as e:
        print(f"❌ Error loading model: {str(e)}")
        return None

def quick_training(use_hyperparameter_tuning=False):
    """
    Quick training function with minimal hyperparameter tuning for faster execution
    """
    print("⚡ QUICK TRAINING MODE")
    print("=" * 40)

    try:
        # Step 1: Load and preprocess data
        X, y, encoders = load_and_preprocess_data()

        # Step 2: Feature engineering
        X_engineered = create_feature_engineering(X)

        # Step 3: Final data type verification
        print("🔍 Final data type check...")
        object_columns = X_engineered.select_dtypes(include=['object']).columns
        if len(object_columns) > 0:
            print(f"❌ Still have object columns: {list(object_columns)}")
            for col in object_columns:
                print(f"🔧 Force converting {col} to numeric")
                X_engineered[col] = pd.to_numeric(X_engineered[col], errors='coerce')
                X_engineered[col] = X_engineered[col].fillna(0)

        print(f"✅ All columns are now numeric: {X_engineered.dtypes.value_counts().to_dict()}")

        # Step 4: Train-test split
        print("📊 Splitting data into train/test sets...")
        X_train, X_test, y_train, y_test = train_test_split(
            X_engineered, y, test_size=0.2, random_state=42, stratify=y
        )
        print(f"Training set: {X_train.shape}, Test set: {X_test.shape}")

        # Step 5: Train model (quick training)
        model = train_xgboost_model(X_train, X_test, y_train, y_test,
                                  hyperparameter_tuning=use_hyperparameter_tuning)

        if model is None:
            print("❌ Model training failed")
            return None, None, None

        # Step 6: Store model info
        store_model_info(model, X_engineered, encoders)

        # Step 7: Evaluate model
        metrics, y_pred, y_pred_proba = evaluate_model(model, X_test, y_test, X_train, y_train)

        # Step 8: Feature analysis (without plots for quick mode)
        importance_df, category_importance = analyze_feature_importance(model, X_engineered.columns, encoders)

        # Step 9: Save model
        save_model(model)

        print(f"\n🎉 QUICK TRAINING COMPLETE!")
        print(f"🎯 Final Test Accuracy: {metrics['Test Accuracy']:.4f}")
        print(f"🎯 Final ROC-AUC: {metrics['ROC-AUC']:.4f}")
        print(f"\n✨ Model ready for predictions!")

        return model, metrics, importance_df

    except Exception as e:
        print(f"❌ Error during execution: {str(e)}")
        import traceback
        traceback.print_exc()
        return None, None, None

def full_training_with_plots():
    """
    Complete training with hyperparameter tuning and visualization
    """
    print("🌙 FULL XGBOOST SLEEP DEPRIVATION TRAINING")
    print("=" * 60)

    try:
        # Step 1: Load and preprocess data
        X, y, encoders = load_and_preprocess_data()

        # Step 2: Feature engineering
        X_engineered = create_feature_engineering(X)

        # Step 3: Final data type verification
        print("🔍 Final data type check...")
        object_columns = X_engineered.select_dtypes(include=['object']).columns
        if len(object_columns) > 0:
            print(f"❌ Still have object columns: {list(object_columns)}")
            for col in object_columns:
                print(f"🔧 Force converting {col} to numeric")
                X_engineered[col] = pd.to_numeric(X_engineered[col], errors='coerce')
                X_engineered[col] = X_engineered[col].fillna(0)

        print(f"✅ All columns are now numeric: {X_engineered.dtypes.value_counts().to_dict()}")

        # Step 4: Train-test split
        print("📊 Splitting data into train/test sets...")
        X_train, X_test, y_train, y_test = train_test_split(
            X_engineered, y, test_size=0.2, random_state=42, stratify=y
        )
        print(f"Training set: {X_train.shape}, Test set: {X_test.shape}")

        # Step 5: Train model with hyperparameter tuning
        model = train_xgboost_model(X_train, X_test, y_train, y_test, hyperparameter_tuning=True)

        if model is None:
            print("❌ Model training failed")
            return None, None, None

        # Step 6: Store model info
        store_model_info(model, X_engineered, encoders)

        # Step 7: Evaluate model
        metrics, y_pred, y_pred_proba = evaluate_model(model, X_test, y_test, X_train, y_train)

        # Step 8: Visualize results
        feature_importance_df = plot_model_analysis(model, X_test, y_test, y_pred_proba, X_engineered.columns)

        # Step 9: Feature analysis
        importance_df, category_importance = analyze_feature_importance(model, X_engineered.columns, encoders)

        # Step 10: Save model
        save_model(model)

        print(f"\n🎉 FULL TRAINING COMPLETE!")
        print(f"🎯 Final Test Accuracy: {metrics['Test Accuracy']:.4f}")
        print(f"🎯 Final ROC-AUC: {metrics['ROC-AUC']:.4f}")
        print(f"\n✨ Model ready for predictions and manual testing!")

        return model, metrics, importance_df

    except Exception as e:
        print(f"❌ Error during execution: {str(e)}")
        import traceback
        traceback.print_exc()
        return None, None, None

# Manual input testing functions
def get_manual_input():
    """
    Function to collect manual input for testing the sleep deprivation model
    """
    print("\n🔍 MANUAL INPUT FOR SLEEP DEPRIVATION PREDICTION")
    print("=" * 60)
    print("Please enter the following information:")

    manual_data = {}

    # Basic demographics
    print("\n👤 DEMOGRAPHICS:")
    manual_data['age'] = float(input("Age (e.g., 28, 40, 65): "))
    manual_data['gender'] = input("Gender (Male/Female): ").strip()

    # Sleep metrics
    print("\n😴 SLEEP INFORMATION:")
    manual_data['sleep_hours'] = float(input("Sleep hours (e.g., 6.4, 7.6, 8.5): "))
    manual_data['sleep_quality'] = int(input("Sleep quality (1-10 scale, e.g., 5, 8, 9): "))

    bedtime_input = input("Bedtime (e.g., 21:57, 22:50, 23:12): ").strip()
    manual_data['bedtime'] = bedtime_input

    wakeup_input = input("Wake up time (e.g., 6:30, 5:15, 7:43): ").strip()
    manual_data['wakeup_time'] = wakeup_input

    manual_data['sleep_latency'] = int(input("Sleep latency in minutes (time to fall asleep, e.g., 7, 31, 2): "))

    # Technology usage
    print("\n📱 TECHNOLOGY USAGE:")
    manual_data['screen_time_hours'] = float(input("Screen time hours (e.g., 1.9, 9.8, 2.5): "))
    manual_data['in_bed_phone_use_percent'] = int(input("In-bed phone use percentage (0-100, e.g., 45, 70, 28): "))
    manual_data['social_media_hours'] = float(input("Social media hours (e.g., 3.6, 5.0, 2.6): "))
    manual_data['light_exposure_before_bed'] = input("Light exposure before bed (Low/Medium/High): ").strip()

    # Health and lifestyle
    print("\n🏃 HEALTH & LIFESTYLE:")
    manual_data['caffeine_intake'] = int(input("Caffeine intake in mg (e.g., 73, 141, 166): "))
    manual_data['physical_activity_mins'] = int(input("Physical activity minutes (e.g., 57, 53, 18): "))
    manual_data['water_intake_liters'] = float(input("Water intake in liters (e.g., 3.2, 2.8, 1.8): "))
    manual_data['stress_level'] = int(input("Stress level (1-10 scale, e.g., 3, 9, 4): "))
    manual_data['energy_level'] = int(input("Energy level (1-10 scale, e.g., 5, 3, 10): "))

    # Diet and timing
    manual_data['diet_meal_timing'] = int(input("Diet meal timing (1-4 scale, e.g., 3, 4, 2): "))

    # Sleep patterns
    print("\n🌙 SLEEP PATTERNS:")
    manual_data['preferred_sleep_time_category'] = input("Preferred sleep time category (Early Bird/Normal Sleeper/Night Owl): ").strip()
    manual_data['sleep_consistency_score'] = int(input("Sleep consistency score (1-10, e.g., 6, 5, 8): "))

    # Work and daily routine
    print("\n💼 WORK & DAILY ROUTINE:")
    manual_data['day_type'] = input("Day type (Weekday/Weekend): ").strip()
    manual_data['occupation_type'] = input("Occupation type (Teacher/Retail Worker/Student/Engineer/etc.): ").strip()
    manual_data['study_or_work_hours'] = float(input("Study/work hours (e.g., 6.6, 5.6, 8.5): "))
    manual_data['work_shift_type'] = input("Work shift type (Day Shift/Night Shift/Flexible): ").strip()
    manual_data['daily_commute_time_mins'] = int(input("Daily commute time in minutes (e.g., 28, 23, 4): "))

    # Additional factors
    print("\n🌞 ADDITIONAL FACTORS:")
    manual_data['afternoon_naps'] = int(input("Afternoon nap duration in minutes (0 if no nap, e.g., 0, 45, 75): "))
    manual_data['time_spent_outdoors_daily'] = int(input("Time spent outdoors daily in minutes (e.g., 159, 131, 64): "))

    # Habits
    print("\n🚬🍷 HABITS:")
    manual_data['smoking'] = input("Smoking habit (Never/Regular/Heavy): ").strip()
    manual_data['alcohol_habit'] = input("Alcohol habit (Never/Occasionally/Moderate/Heavy): ").strip()

    # Environment and health
    print("\n🏠 ENVIRONMENT & HEALTH:")
    manual_data['sleep_environment_quality'] = input("Sleep environment quality (Poor/Fair/Good/Excellent): ").strip()
    manual_data['medical_conditions'] = input("Medical conditions (None/Hypertension/Diabetes/etc.): ").strip()

    # Age group (derived from age)
    if manual_data['age'] <= 25:
        manual_data['age_group'] = '18-25'
    elif manual_data['age'] <= 35:
        manual_data['age_group'] = '26-35'
    elif manual_data['age'] <= 50:
        manual_data['age_group'] = '36-50'
    else:
        manual_data['age_group'] = '51-65'

    return manual_data

def preprocess_manual_input(manual_data, label_encoders):
    """
    Preprocess manual input to match the training data format
    """
    print("\n🔄 Preprocessing manual input...")

    # Create a DataFrame with manual input
    manual_df = pd.DataFrame([manual_data])

    # Handle time features
    if 'bedtime' in manual_df.columns:
        def time_to_minutes(time_str):
            try:
                if pd.isna(time_str):
                    return 0
                time_str = str(time_str).strip()
                if ':' in time_str:
                    hours, minutes = map(int, time_str.split(':'))
                    return hours * 60 + minutes
                else:
                    return float(time_str)
            except:
                return 0

        manual_df['bedtime_minutes'] = manual_df['bedtime'].apply(time_to_minutes)
        manual_df = manual_df.drop('bedtime', axis=1)

    if 'wakeup_time' in manual_df.columns:
        manual_df['wakeup_minutes'] = manual_df['wakeup_time'].apply(time_to_minutes)
        manual_df = manual_df.drop('wakeup_time', axis=1)

    # Apply label encoding for categorical variables
    for col in manual_df.columns:
        if col in label_encoders and manual_df[col].dtype == 'object':
            le = label_encoders[col]
            try:
                manual_df[col] = le.transform(manual_df[col].astype(str))
            except ValueError:
                print(f"⚠️  Unknown value for {col}: {manual_df[col].iloc[0]}. Using default encoding.")
                manual_df[col] = 0

    # Convert any remaining object columns to numeric
    for col in manual_df.columns:
        if manual_df[col].dtype == 'object':
            try:
                manual_df[col] = pd.to_numeric(manual_df[col], errors='coerce')
            except:
                manual_df[col] = 0

    # Fill any NaN values
    manual_df = manual_df.fillna(0)

    # Ensure all columns are numeric
    for col in manual_df.columns:
        if manual_df[col].dtype not in ['int8', 'int16', 'int32', 'int64', 'float16', 'float32', 'float64', 'bool']:
            manual_df[col] = manual_df[col].astype('float64')

    return manual_df

def create_manual_features(manual_df):
    """
    Apply feature engineering to manual input
    """
    print("🔧 Creating engineered features for manual input...")

    manual_eng = manual_df.copy()

    # Sleep efficiency features
    if 'sleep_hours' in manual_eng.columns and 'sleep_latency' in manual_eng.columns:
        manual_eng['sleep_efficiency'] = manual_eng['sleep_hours'] / (manual_eng['sleep_hours'] + manual_eng['sleep_latency']/60 + 1e-6)

    # Technology usage intensity
    if 'screen_time_hours' in manual_eng.columns and 'social_media_hours' in manual_eng.columns:
        manual_eng['total_screen_time'] = manual_eng['screen_time_hours'] + manual_eng['social_media_hours']

    if 'in_bed_phone_use_percent' in manual_eng.columns and 'total_screen_time' in manual_eng.columns:
        manual_eng['digital_dependency'] = (manual_eng['in_bed_phone_use_percent'] / 100) * manual_eng['total_screen_time']

    # Health & lifestyle score
    health_components = []
    if 'physical_activity_mins' in manual_eng.columns:
        health_components.append(manual_eng['physical_activity_mins'] / 60)
    if 'water_intake_liters' in manual_eng.columns:
        health_components.append(manual_eng['water_intake_liters'] / 3)
    if 'time_spent_outdoors_daily' in manual_eng.columns:
        health_components.append(manual_eng['time_spent_outdoors_daily'] / 120)
    if 'stress_level' in manual_eng.columns:
        health_components.append(-manual_eng['stress_level'] / 10)
    if 'caffeine_intake' in manual_eng.columns:
        health_components.append(-manual_eng['caffeine_intake'] / 400)

    if health_components:
        manual_eng['health_score'] = sum(health_components)

    # Sleep quality composite
    sleep_quality_components = []
    if 'sleep_quality' in manual_eng.columns:
        sleep_quality_components.append(manual_eng['sleep_quality'])
    if 'sleep_consistency_score' in manual_eng.columns:
        sleep_quality_components.append(manual_eng['sleep_consistency_score'])
    if 'sleep_latency' in manual_eng.columns:
        sleep_quality_components.append(-manual_eng['sleep_latency'] / 30)

    if sleep_quality_components:
        manual_eng['sleep_quality_composite'] = sum(sleep_quality_components) / len(sleep_quality_components)

    # Work-life balance indicator
    work_life_components = []
    if 'study_or_work_hours' in manual_eng.columns:
        work_life_components.append(manual_eng['study_or_work_hours'])
    if 'daily_commute_time_mins' in manual_eng.columns:
        work_life_components.append(manual_eng['daily_commute_time_mins']/60)

    if work_life_components:
        manual_eng['work_life_balance'] = sum(work_life_components)

    # Ensure all features are numeric
    for col in manual_eng.columns:
        if manual_eng[col].dtype not in ['int8', 'int16', 'int32', 'int64', 'float16', 'float32', 'float64', 'bool']:
            manual_eng[col] = manual_eng[col].astype('float64')

    return manual_eng

def predict_manual_input(model, manual_input_processed, feature_columns):
    """
    Make prediction on manual input
    """
    print("\n🔮 Making prediction...")

    # Ensure manual input has all required columns in the same order
    missing_cols = set(feature_columns) - set(manual_input_processed.columns)
    if missing_cols:
        print(f"⚠️  Adding missing columns with default values: {missing_cols}")
        for col in missing_cols:
            manual_input_processed[col] = 0

    # Reorder columns to match training data
    manual_input_processed = manual_input_processed[feature_columns]

    # Make prediction
    prediction = model.predict(manual_input_processed)[0]
    prediction_proba = model.predict_proba(manual_input_processed)[0]

    print(f"\n🎯 PREDICTION RESULTS:")
    print("=" * 40)
    print(f"Sleep Deprivation Prediction: {'YES' if prediction == 1 else 'NO'}")
    print(f"Confidence (No Sleep Deprivation): {prediction_proba[0]:.3f} ({prediction_proba[0]*100:.1f}%)")
    print(f"Confidence (Sleep Deprivation): {prediction_proba[1]:.3f} ({prediction_proba[1]*100:.1f}%)")

    # Risk assessment
    risk_level = "Low" if prediction_proba[1] < 0.3 else "Medium" if prediction_proba[1] < 0.7 else "High"
    print(f"Risk Level: {risk_level}")

    # Recommendations based on prediction
    if prediction == 1:
        print(f"\n💡 RECOMMENDATIONS:")
        print("- Consider improving sleep hygiene")
        print("- Reduce screen time before bed")
        print("- Maintain consistent sleep schedule")
        print("- Manage stress levels")
        print("- Limit caffeine intake")

    return prediction, prediction_proba

def test_with_manual_input():
    """
    Main function to test the trained model with manual input
    """
    global trained_model, feature_columns, label_encoders

    if trained_model is None:
        print("❌ No trained model found. Please run training first.")
        print("Use: quick_training() or full_training_with_plots()")
        return

    print("\n🧪 TESTING MODEL WITH MANUAL INPUT")
    print("=" * 50)

    # Get manual input
    manual_data = get_manual_input()

    # Preprocess manual input
    manual_processed = preprocess_manual_input(manual_data, label_encoders)

    # Apply feature engineering
    manual_engineered = create_manual_features(manual_processed)

    # Make prediction
    prediction, prediction_proba = predict_manual_input(trained_model, manual_engineered, feature_columns)

    return prediction, prediction_proba

def create_sample_dataset():
    """
    Create a sample dataset for testing if you don't have the actual CSV file
    """
    print("🔧 Creating sample dataset for testing...")

    np.random.seed(42)
    n_samples = 1000

    # Generate sample data
    data = {
        'age': np.random.randint(18, 66, n_samples),
        'gender': np.random.choice(['Male', 'Female'], n_samples),
        'sleep_hours': np.random.normal(7.5, 1.5, n_samples).clip(4, 12),
        'sleep_quality': np.random.randint(1, 11, n_samples),
        'sleep_latency': np.random.randint(1, 61, n_samples),
        'screen_time_hours': np.random.normal(5, 3, n_samples).clip(0, 16),
        'in_bed_phone_use_percent': np.random.randint(0, 101, n_samples),
        'social_media_hours': np.random.normal(3, 2, n_samples).clip(0, 12),
        'light_exposure_before_bed': np.random.choice(['Low', 'Medium', 'High'], n_samples),
        'caffeine_intake': np.random.randint(0, 401, n_samples),
        'physical_activity_mins': np.random.randint(0, 181, n_samples),
        'water_intake_liters': np.random.normal(2.5, 1, n_samples).clip(0.5, 5),
        'stress_level': np.random.randint(1, 11, n_samples),
        'energy_level': np.random.randint(1, 11, n_samples),
        'diet_meal_timing': np.random.randint(1, 5, n_samples),
        'preferred_sleep_time_category': np.random.choice(['Early Bird', 'Normal Sleeper', 'Night Owl'], n_samples),
        'sleep_consistency_score': np.random.randint(1, 11, n_samples),
        'day_type': np.random.choice(['Weekday', 'Weekend'], n_samples),
        'occupation_type': np.random.choice(['Teacher', 'Student', 'Engineer', 'Retail Worker', 'Healthcare'], n_samples),
        'study_or_work_hours': np.random.normal(7, 2, n_samples).clip(2, 14),
        'work_shift_type': np.random.choice(['Day Shift', 'Night Shift', 'Flexible'], n_samples),
        'daily_commute_time_mins': np.random.randint(0, 121, n_samples),
        'afternoon_naps': np.random.choice([0, 30, 45, 60, 90], n_samples),
        'time_spent_outdoors_daily': np.random.randint(0, 301, n_samples),
        'smoking': np.random.choice(['Never', 'Regular', 'Heavy'], n_samples),
        'alcohol_habit': np.random.choice(['Never', 'Occasionally', 'Moderate', 'Heavy'], n_samples),
        'sleep_environment_quality': np.random.choice(['Poor', 'Fair', 'Good', 'Excellent'], n_samples),
        'medical_conditions': np.random.choice(['None', 'Hypertension', 'Diabetes', 'Anxiety'], n_samples)
    }

    # Create age groups
    ages = data['age']
    age_groups = []
    for age in ages:
        if age <= 25:
            age_groups.append('18-25')
        elif age <= 35:
            age_groups.append('26-35')
        elif age <= 50:
            age_groups.append('36-50')
        else:
            age_groups.append('51-65')

    data['age_group'] = age_groups

    # Generate target variable based on some logical rules
    sleep_deprivation = []
    for i in range(n_samples):
        score = 0
        # Poor sleep hours
        if data['sleep_hours'][i] < 6:
            score += 0.3
        # High stress
        if data['stress_level'][i] > 7:
            score += 0.2
        # High screen time
        if data['screen_time_hours'][i] > 8:
            score += 0.2
        # High caffeine
        if data['caffeine_intake'][i] > 200:
            score += 0.15
        # Poor sleep quality
        if data['sleep_quality'][i] < 5:
            score += 0.2
        # High sleep latency
        if data['sleep_latency'][i] > 30:
            score += 0.15

        # Add some randomness
        score += np.random.normal(0, 0.1)

        sleep_deprivation.append(1 if score > 0.5 else 0)

    data['sleep_deprivation'] = sleep_deprivation

    # Create DataFrame and save
    df = pd.DataFrame(data)
    df.to_csv('sample_sleep_dataset.csv', index=False)

    print(f"✅ Sample dataset created with {n_samples} records")
    print(f"📊 Sleep deprivation rate: {np.mean(sleep_deprivation)*100:.1f}%")

    return df

# MAIN EXECUTION FUNCTIONS
def run_quick_training():
    """
    Run quick training without hyperparameter tuning
    """
    print("⚡ STARTING QUICK TRAINING...")
    model, metrics, importance = quick_training(use_hyperparameter_tuning=False)

    if model is not None:
        print("\n🎉 Quick training successful!")
        print("✨ You can now test with: test_with_manual_input()")

    return model, metrics, importance

def run_full_training():
    """
    Run full training with hyperparameter tuning and plots
    """
    print("🚀 STARTING FULL TRAINING...")
    model, metrics, importance = full_training_with_plots()

    if model is not None:
        print("\n🎉 Full training successful!")
        print("✨ You can now test with: test_with_manual_input()")

    return model, metrics, importance

def run_training_with_sample_data():
    """
    Create sample data and run training (useful for testing)
    """
    print("🧪 CREATING SAMPLE DATA AND TRAINING...")

    # Create sample dataset
    sample_df = create_sample_dataset()

    # Run quick training
    model, metrics, importance = quick_training(use_hyperparameter_tuning=False)

    return model, metrics, importance

# USAGE INSTRUCTIONS AND EXAMPLES
def print_usage_instructions():
    """
    Print clear usage instructions
    """
    print("\n📚 USAGE INSTRUCTIONS:")
    print("=" * 60)
    print("1. QUICK TRAINING (Recommended for testing):")
    print("   model, metrics, importance = run_quick_training()")
    print()
    print("2. FULL TRAINING (With hyperparameter tuning and plots):")
    print("   model, metrics, importance = run_full_training()")
    print()
    print("3. TRAINING WITH SAMPLE DATA (If you don't have CSV file):")
    print("   model, metrics, importance = run_training_with_sample_data()")
    print()
    print("4. MANUAL TESTING (After training):")
    print("   prediction, probabilities = test_with_manual_input()")
    print()
    print("5. LOAD SAVED MODEL:")
    print("   model = load_model('sleep_deprivation_model.pkl')")
    print()
    print("💡 TIPS:")
    print("- Start with run_quick_training() for fastest results")
    print("- Use run_full_training() for best model performance")
    print("- Make sure 'organized_sleep_dataset.csv' is in your directory")
    print("- Or use run_training_with_sample_data() to create test data")

if __name__ == "__main__":
    # Print usage instructions
    print_usage_instructions()

    # Uncomment one of these to run training:

    # Option 1: Quick training (fastest)
    # model, metrics, importance = run_quick_training()

    # Option 2: Full training with tuning
    model, metrics, importance = run_full_training()

    # Option 3: Training with sample data
    # model, metrics, importance = run_training_with_sample_data()

    print("\n🎯 To start training, uncomment one of the options above or run:")
    print("model, metrics, importance = run_quick_training()")