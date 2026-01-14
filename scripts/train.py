import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, RandomizedSearchCV
from lightgbm import LGBMRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error, make_scorer
from sklearn.feature_selection import SelectKBest, f_regression
import joblib
from pathlib import Path
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

SCRIPT_DIR = Path(__file__).parent
PROJECT_ROOT = SCRIPT_DIR.parent
DATA_CLEANED_PATH = PROJECT_ROOT / "data" / "cleaned" / "student_performance_cleaned.csv"
MODELS_DIR = PROJECT_ROOT / "models"

MODELS_DIR.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(DATA_CLEANED_PATH)

print(f"Data shape: {df.shape}")

df = df.dropna(subset=['Exam_Score'])

"""
IQR method for outlier removal
"""
Q1 = df['Exam_Score'].quantile(0.25)
Q3 = df['Exam_Score'].quantile(0.75)
IQR = Q3 - Q1
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR
outliers_count = ((df['Exam_Score'] < lower_bound) | (df['Exam_Score'] > upper_bound)).sum()
print(f"\nOutliers detected in Exam_Score: {outliers_count} ({outliers_count/len(df)*100:.2f}%)")
df = df[(df['Exam_Score'] >= lower_bound) & (df['Exam_Score'] <= upper_bound)]
print(f"Data shape after outlier removal: {df.shape}")

features = [col for col in df.columns if col != 'Exam_Score']
X = df[features]
y = df['Exam_Score']

print(f"\nFeatures used: {len(features)}")

print("\nPerforming feature selection...")
selector = SelectKBest(score_func=f_regression, k='all')
selector.fit(X, y)
feature_scores = pd.DataFrame({
    'feature': features,
    'score': selector.scores_
}).sort_values('score', ascending=False)

min_score_threshold = 1.0
selected_features = feature_scores[feature_scores['score'] > min_score_threshold]['feature'].tolist()
if len(selected_features) < len(features):
    print(f"Selected {len(selected_features)} features out of {len(features)}")
    print("Top 10 features by importance:")
    print(feature_scores.head(10).to_string(index=False))
    features = selected_features
    X = X[features]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(f"\nTraining set size: {X_train.shape[0]}")
print(f"Test set size: {X_test.shape[0]}")
print(f"Final number of features: {len(features)}")

print("\n" + "="*50)
print("Hyperparameter Tuning")
print("="*50)

"""
LGBMRegressor hyperparameters for randomized search
"""
param_grid = {
    'n_estimators': [100, 200, 300, 500],
    'max_depth': [3, 5, 7, 10, -1],
    'learning_rate': [0.01, 0.05, 0.1, 0.15],
    'num_leaves': [15, 31, 50, 100],
    'min_child_samples': [10, 20, 30],
    'subsample': [0.8, 0.9, 1.0],
    'colsample_bytree': [0.8, 0.9, 1.0],
    'reg_alpha': [0, 0.1, 0.5],
    'reg_lambda': [0, 0.1, 0.5, 1.0]
}

base_model = LGBMRegressor(random_state=42, n_jobs=-1, verbose=-1)

"""
Randomized search with 5-fold cross-validation
"""
print("Performing randomized search with 5-fold cross-validation...")
print("This may take a few minutes...")
random_search = RandomizedSearchCV(
    estimator=base_model,
    param_distributions=param_grid,
    n_iter=50,
    cv=5,
    scoring='neg_mean_squared_error',
    n_jobs=-1,
    random_state=42,
    verbose=1
)

random_search.fit(X_train, y_train)

print(f"\nBest parameters found:")
for param, value in random_search.best_params_.items():
    print(f"  {param}: {value}")

model = random_search.best_estimator_

print("\n" + "="*50)
print("Cross-Validation Results (5-fold)")
print("="*50)
cv_scores_rmse = cross_val_score(model, X_train, y_train, cv=5, 
                                  scoring=make_scorer(lambda y_true, y_pred: np.sqrt(mean_squared_error(y_true, y_pred))))
cv_scores_r2 = cross_val_score(model, X_train, y_train, cv=5, scoring='r2')
print(f"RMSE: {cv_scores_rmse.mean():.4f} (+/- {cv_scores_rmse.std() * 2:.4f})")
print(f"R²: {cv_scores_r2.mean():.4f} (+/- {cv_scores_r2.std() * 2:.4f})")

model.fit(X_train, y_train)

y_pred = model.predict(X_test)

mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("\n" + "="*50)
print("Test Set Evaluation Results:")
print("="*50)
print(f'Mean Squared Error (MSE): {mse:.2f}')
print(f'Root Mean Squared Error (RMSE): {rmse:.2f}')
print(f'Mean Absolute Error (MAE): {mae:.2f}')
print(f'R² Score: {r2:.4f}')
print("="*50)

print("\n" + "="*50)
print("Top 10 Most Important Features:")
print("="*50)
feature_importance = pd.DataFrame({
    'feature': features,
    'importance': model.feature_importances_
}).sort_values('importance', ascending=False)
print(feature_importance.head(10).to_string(index=False))
print("="*50)

timestamp = datetime.now().strftime("%d-%m-%Y_%H-%M-%S")
MODEL_PATH = MODELS_DIR / f"model_{timestamp}.pkl"
joblib.dump(model, MODEL_PATH)
print(f"\nModel saved to: {MODEL_PATH}")

import json
feature_info = {
    'features': features,
    'target': 'Exam_Score',
    'best_params': random_search.best_params_,
    'test_metrics': {
        'mse': float(mse),
        'rmse': float(rmse),
        'mae': float(mae),
        'r2': float(r2)
    },
    'cv_metrics': {
        'rmse_mean': float(cv_scores_rmse.mean()),
        'rmse_std': float(cv_scores_rmse.std()),
        'r2_mean': float(cv_scores_r2.mean()),
        'r2_std': float(cv_scores_r2.std())
    },
    'feature_importance': feature_importance.to_dict('records')
}

FEATURE_INFO_PATH = PROJECT_ROOT / "models" / "feature_info.json"
with open(FEATURE_INFO_PATH, 'w') as f:
    json.dump(feature_info, f, indent=2)
print(f"\nFeature information and model metadata saved to: {FEATURE_INFO_PATH}")

