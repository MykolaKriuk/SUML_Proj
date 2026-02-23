# Student Performance Prediction — Machine Learning Project

A machine learning project that predicts student exam scores from factors such as study habits, school environment, family background, and personal characteristics. Built with LightGBM and a Streamlit web interface for exploration and prediction.

**Live app:** [mmk-student-performance.streamlit.app](https://mmk-student-performance.streamlit.app/)

---

## Table of Contents

- [Overview](#overview)
- [Dataset](#dataset)
- [Project Structure](#project-structure)
- [Setup](#setup)
- [Usage](#usage)
- [Pipeline](#pipeline)
- [Model](#model)
- [Web Application](#web-application)
- [Scripts Reference](#scripts-reference)

---

## Overview

This project:

1. **Cleans and engineers features** from raw student performance data
2. **Trains a LightGBM regressor** with hyperparameter tuning
3. **Exposes a Streamlit app** for data exploration, model inspection, and predictions

**Target variable:** `Exam_Score` (continuous, 0–100)

---

## Dataset

| Property | Value |
|----------|-------|
| **Source** | [Kaggle — Student Performance Factors](https://www.kaggle.com/datasets/lainguyn123/student-performance-factors) |
| **Records** | 6,607 |
| **Original features** | 20 |

### Original Features

| Category | Features |
|----------|----------|
| **Study** | Hours_Studied, Sleep_Hours, Tutoring_Sessions |
| **School** | Attendance, Teacher_Quality, School_Type |
| **Personal** | Gender, Motivation_Level, Physical_Activity, Learning_Disabilities |
| **Family** | Parental_Involvement, Family_Income, Parental_Education_Level |
| **Environment** | Internet_Access, Access_to_Resources, Extracurricular_Activities, Peer_Influence, Distance_from_Home |

---

## Project Structure

```
SUML_Proj/
├── data/
│   ├── raw/
│   │   └── StudentPerformanceFactors.csv    # Original dataset
│   ├── cleaned/
│   │   └── student_performance_cleaned.csv # Processed + engineered features
│   └── images/                              # EDA plots (correlation, distributions)
│       ├── correlation_matrix.png
│       ├── top_correlations.png
│       └── exam_score_distribution.png
├── models/
│   ├── model_14-01-2026_14-40-49.pkl        # Trained LightGBM model
│   └── feature_info.json                    # Features, metrics, hyperparameters
├── scripts/
│   ├── 01_eda.py                            # EDA, cleaning, feature engineering
│   ├── train.py                             # Model training and evaluation
│   ├── test_best_worst.py                   # Best/worst prediction analysis
│   └── ui/
│       └── app.py                           # Streamlit web app
├── requirements.txt
└── README.md
```

---

## Setup

### Prerequisites

- Python 3.10+
- pip

### Installation

```bash
# Clone or navigate to the project
cd SUML_Proj

# Create virtual environment (recommended)
python -m venv .venv
source .venv/bin/activate   # Linux/macOS
# .venv\Scripts\activate    # Windows

# Install dependencies
pip install -r requirements.txt
```

### Dependencies

| Package | Purpose |
|---------|---------|
| streamlit | Web UI |
| pandas | Data handling |
| numpy | Numerical operations |
| scikit-learn | Preprocessing, metrics, feature selection |
| lightgbm | Gradient boosting model |
| joblib | Model serialization |
| seaborn, matplotlib | EDA plots (used in `01_eda.py`) |

---

## Usage

### 1. Prepare data and run EDA

```bash
python scripts/01_eda.py
```

- Loads raw data from `data/raw/StudentPerformanceFactors.csv`
- Handles missing values, encodes categoricals, engineers features
- Saves cleaned data to `data/cleaned/student_performance_cleaned.csv`
- Generates plots in `data/images/`

### 2. Train the model

```bash
python scripts/train.py
```

- Loads cleaned data
- Removes outliers (IQR on `Exam_Score`)
- Runs feature selection (SelectKBest)
- Performs RandomizedSearchCV for hyperparameter tuning
- Saves model to `models/model_<timestamp>.pkl`
- Updates `models/feature_info.json`

### 3. Run the web app

```bash
streamlit run scripts/ui/app.py
```

Opens the app in the browser (default: http://localhost:8501).

## Pipeline

### Data cleaning (`01_eda.py`)

1. **Missing values:** Fill `Parental_Education_Level`, `Teacher_Quality`, `Distance_from_Home` with `"Unknown"`.
2. **Encoding:**
   - Binary: Gender, Internet_Access, Extracurricular_Activities, Learning_Disabilities
   - Ordinal: remaining categoricals via `LabelEncoder`
3. **Feature engineering:**
   - `study_efficiency` = Hours_Studied / Sleep_Hours
   - `effective_study_hours` = Hours_Studied + 2 × Tutoring_Sessions
   - `attendance_study_interaction`, `previous_study_interaction`, `motivation_study_interaction`
   - `study_sleep_ratio`, `tutoring_study_ratio`
   - `parental_support_score`, `school_quality_score`
   - Squared terms for Hours_Studied, Attendance, Previous_Scores, Sleep_Hours
   - `study_hours_category`, `sleep_category` (binned)

### Training pipeline (`train.py`)

1. Drop rows with missing `Exam_Score`
2. Remove outliers using IQR on `Exam_Score`
3. Feature selection with `SelectKBest` (f_regression, score > 1.0)
4. Train/test split (80/20, `random_state=42`)
5. Hyperparameter search: `RandomizedSearchCV`, 50 iterations, 5-fold CV, neg MSE
6. Final model fit and evaluation (MSE, RMSE, MAE, R²)

---

## Model

### Algorithm

**LightGBM Regressor** with tuned hyperparameters.

### Example metrics (from `feature_info.json`)

| Metric | Test | Cross-validation |
|--------|------|------------------|
| RMSE | 0.42 | 0.44 ± 0.01 |
| MAE | 0.34 | — |
| R² | 0.98 | 0.98 ± 0.00 |
| MSE | 0.18 | — |

### Top features (by importance)

1. Attendance  
2. attendance_study_interaction  
3. previous_study_interaction  
4. Previous_Scores  
5. Access_to_Resources  
6. Parental_Involvement  
7. Parental_Education_Level  
8. Physical_Activity  
9. effective_study_hours  
10. Tutoring_Sessions  

### Model file

The app uses `models/model_14-01-2026_14-40-49.pkl`. To use a different model, update `MODEL_PATH` in `scripts/ui/app.py`.

---

## Web Application

The app is deployed on Streamlit Cloud: **[mmk-student-performance.streamlit.app](https://mmk-student-performance.streamlit.app/)**

### Pages

| Page | Description |
|------|-------------|
| **Home** | Overview, dataset info, summary metrics |
| **Data Overview** | Column info, statistics, target summary |
| **Exploratory Data Analysis** | Correlation matrix, top correlations, exam score distribution |
| **Model Results** | Metrics, hyperparameters, feature importance |
| **Predictions** | Form to input student features and get predicted exam score |

### Prediction form

Inputs include:

- Numeric: Hours Studied, Attendance, Sleep Hours, Previous Scores, Tutoring Sessions, Physical Activity
- Categorical: Motivation Level, Internet Access, Extracurricular Activities, Learning Disabilities, Gender, Parental Involvement, Access to Resources, Peer Influence, Family Income, Teacher Quality, School Type, Parental Education Level, Distance from Home

The app applies the same feature engineering as training before prediction.

### Running the app

```bash
streamlit run scripts/ui/app.py
```

Ensure `lightgbm` is installed; otherwise loading the model will fail with `No module named 'lightgbm'`.

---

## Scripts Reference

### `scripts/01_eda.py`

- Input: `data/raw/StudentPerformanceFactors.csv`
- Output: `data/cleaned/student_performance_cleaned.csv`, `data/images/*.png`
- Steps: load, clean, encode, engineer features, save, generate plots

### `scripts/train.py`

- Input: `data/cleaned/student_performance_cleaned.csv`
- Output: `models/model_<timestamp>.pkl`, `models/feature_info.json`
- Steps: load, outlier removal, feature selection, train/test split, hyperparameter search, fit, save model and metadata

### `scripts/test_best_worst.py`

- Input: `data/cleaned/student_performance_cleaned.csv`, `models/model_14-01-2026_14-40-49.pkl`, `models/feature_info.json`
- Output: prints best and worst prediction cases to stdout
- Steps: load data, model, apply same outlier filtering as training, predict, find min/max absolute error rows

### `scripts/ui/app.py`

- Input: cleaned data, model, `feature_info.json`
- Output: Streamlit web interface
- Paths: `MODEL_PATH` and `FEATURE_INFO_PATH` point to the latest model and metadata

---

## License

For academic use (SUML course project).
