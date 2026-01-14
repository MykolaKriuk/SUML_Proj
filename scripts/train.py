import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import joblib
from pathlib import Path

SCRIPT_DIR = Path(__file__).parent
PROJECT_ROOT = SCRIPT_DIR.parent
DATA_CLEANED_PATH = PROJECT_ROOT / "data" / "cleaned" / "student_performance_cleaned.csv"
MODEL_PATH = PROJECT_ROOT / "models" / "student_performance_model.pkl"

MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)

# Loading data
df = pd.read_csv(DATA_CLEANED_PATH)

print(f"Data shape: {df.shape}")

# Remove records with missing values in the target column
df = df.dropna(subset=['Exam_Score'])

# Select features and target
# Exclude Exam_Score from features
features = [col for col in df.columns if col != 'Exam_Score']
X = df[features]
y = df['Exam_Score']

print(f"\nFeatures used: {len(features)}")

# Split into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(f"\nTraining set size: {X_train.shape[0]}")
print(f"Test set size: {X_test.shape[0]}")

# Training model
# 3
# model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)

# 2
# model = XGBRegressor(n_estimators=100, random_state=42, n_jobs=-1)

# 1 -> best
model = LGBMRegressor(n_estimators=100, random_state=42, n_jobs=-1)
model.fit(X_train, y_train)

# Model evaluation
y_pred = model.predict(X_test)

# Regression metrics
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("\n" + "="*50)
print("Model Evaluation Results:")
print("="*50)
print(f'Mean Squared Error (MSE): {mse:.2f}')
print(f'Root Mean Squared Error (RMSE): {rmse:.2f}')
print(f'Mean Absolute Error (MAE): {mae:.2f}')
print(f'R² Score: {r2:.4f}')
print("="*50)

# Save model
joblib.dump(model, MODEL_PATH)
print(f"\nModel saved to: {MODEL_PATH}")

# Save feature information (for later use in prediction)
feature_info = {
    'features': features,
    'target': 'Exam_Score'
}
import json
FEATURE_INFO_PATH = PROJECT_ROOT / "models" / "feature_info.json"
with open(FEATURE_INFO_PATH, 'w') as f:
    json.dump(feature_info, f, indent=2)
print(f"Feature information saved to: {FEATURE_INFO_PATH}")

