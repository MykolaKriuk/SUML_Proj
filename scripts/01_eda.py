import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from pathlib import Path


SCRIPT_DIR = Path(__file__).parent
PROJECT_ROOT = SCRIPT_DIR.parent
DATA_RAW_PATH = PROJECT_ROOT / "data" / "raw" / "StudentPerformanceFactors.csv"
DATA_CLEANED_PATH = PROJECT_ROOT / "data" / "cleaned" / "student_performance_cleaned.csv"

df = pd.read_csv(DATA_RAW_PATH)

df.head() 
df.info()  
df.describe()  
df.columns  
print(df.isnull().sum())

num_cols = df.select_dtypes(include=['int64', 'float64']).columns
cat_cols = df.select_dtypes(include=['object', 'category']).columns

for col in ["Parental_Education_Level", "Teacher_Quality", "Distance_from_Home"]:
	df[col] = df[col].fillna("Unknown")

print()
print("After adding unknown:")
print(df.isnull().sum())

print("\nCreating new features...")

df["study_efficiency"] = df["Hours_Studied"] / df["Sleep_Hours"].replace(0, 0.5)
df["effective_study_hours"] = df["Hours_Studied"] + 2 * df["Tutoring_Sessions"]

if 'Attendance' in df.columns and 'Hours_Studied' in df.columns:
    df["attendance_study_interaction"] = df["Attendance"] * df["Hours_Studied"] / 100
    
if 'Previous_Scores' in df.columns and 'Hours_Studied' in df.columns:
    df["previous_study_interaction"] = df["Previous_Scores"] * df["Hours_Studied"] / 100
    
if 'Motivation_Level' in df.columns and 'Hours_Studied' in df.columns:
    df["motivation_study_interaction"] = df["Motivation_Level"] * df["Hours_Studied"]

if 'Hours_Studied' in df.columns and 'Sleep_Hours' in df.columns:
    df["study_sleep_ratio"] = df["Hours_Studied"] / (df["Sleep_Hours"] + 1)
    
if 'Tutoring_Sessions' in df.columns and 'Hours_Studied' in df.columns:
    df["tutoring_study_ratio"] = df["Tutoring_Sessions"] / (df["Hours_Studied"] + 1)

if 'Parental_Involvement' in df.columns and 'Parental_Education_Level' in df.columns:
    pass

print(f"Total features after engineering: {len(df.columns)}")
binary_maps = {
	"Gender": {"Male": 0, "Female": 1},
	"Internet_Access": {"No": 0, "Yes": 1},
	"Extracurricular_Activities": {"No": 0, "Yes": 1},
	"Learning_Disabilities": {"No": 0, "Yes": 1},
}
for col, mapping in binary_maps.items():
	df[col] = df[col].map(mapping)

binary_cols = set(binary_maps.keys())
remaining_cat_cols = [col for col in cat_cols if col not in binary_cols]

for col in remaining_cat_cols:
	label_enc = LabelEncoder()
	df[col] = label_enc.fit_transform(df[col])

print("\nCreating additional interaction features after encoding...")

if 'Parental_Involvement' in df.columns and 'Parental_Education_Level' in df.columns:
    df["parental_support_score"] = df["Parental_Involvement"] * (df["Parental_Education_Level"] + 1)

if 'Teacher_Quality' in df.columns and 'School_Type' in df.columns:
    df["school_quality_score"] = df["Teacher_Quality"] * (df["School_Type"] + 1)

key_features = ['Hours_Studied', 'Attendance', 'Previous_Scores', 'Sleep_Hours']
for feature in key_features:
    if feature in df.columns:
        df[f"{feature}_squared"] = df[feature] ** 2

if 'Hours_Studied' in df.columns:
    df["study_hours_category"] = pd.cut(df["Hours_Studied"], bins=5, labels=False, duplicates='drop')
    df["study_hours_category"] = df["study_hours_category"].astype(float)
    
if 'Sleep_Hours' in df.columns:
    df["sleep_category"] = pd.cut(df["Sleep_Hours"], bins=5, labels=False, duplicates='drop')
    df["sleep_category"] = df["sleep_category"].astype(float)

df = df.fillna(0)

print("\nEnsuring all columns are numeric...")
non_numeric_cols = []
for col in df.columns:
    if col != 'Exam_Score':
        if not pd.api.types.is_numeric_dtype(df[col]):
            non_numeric_cols.append(col)
            try:
                df[col] = pd.to_numeric(df[col], errors='coerce')
                df[col] = df[col].fillna(0)
                print(f"  Converted {col} to numeric")
            except:
                print(f"  Warning: Could not convert {col} to numeric, will be dropped")
                df = df.drop(columns=[col])

if non_numeric_cols:
    print(f"Handled {len(non_numeric_cols)} non-numeric columns")

print(f"Final total features: {len(df.columns)}")
print(f"Data types: {df.dtypes.value_counts().to_dict()}")
print(df.head())
print(df["Gender"])
print(df["Motivation_Level"])

df.to_csv(DATA_CLEANED_PATH, index=False)

IMAGES_DIR = PROJECT_ROOT / "data" / "images"
CORRELATION_IMAGE_PATH = IMAGES_DIR / "correlation_matrix.png"
TOP_CORRELATIONS_IMAGE_PATH = IMAGES_DIR / "top_correlations.png"
EXAM_SCORE_DISTRIBUTION_IMAGE_PATH = IMAGES_DIR / "exam_score_distribution.png"

IMAGES_DIR.mkdir(parents=True, exist_ok=True)

corr = df.corr(numeric_only=True)

corr_unstacked = corr.abs().unstack().sort_values(ascending=False)
print("\nTop correlated pairs:")
print(corr_unstacked[(corr_unstacked < 1) & (corr_unstacked > 0.7)])

plt.figure(figsize=(16, 12))
sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f", square=True)
plt.title("Correlation Matrix of Student Performance Features")
plt.savefig(CORRELATION_IMAGE_PATH)
plt.close()

threshold = 0.85
corr_matrix = df.corr(numeric_only=True).abs()
upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))

to_drop = [column for column in upper.columns if any(upper[column] > threshold)]
print("\nHighly correlated columns to drop (threshold > 0.85):", to_drop)

if 'Exam_Score' in df.columns:
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    correlations = df[numeric_cols].corr()['Exam_Score'].abs().sort_values(ascending=False)
    correlations = correlations[correlations.index != 'Exam_Score']
    
    fig, ax = plt.subplots(figsize=(10, 6))
    top_10 = correlations.head(10)
    ax.barh(range(len(top_10)), top_10.values)
    ax.set_yticks(range(len(top_10)))
    ax.set_yticklabels(top_10.index)
    ax.set_xlabel('Absolute Correlation with Exam Score')
    ax.set_title('Top 10 Features Correlated with Exam Score')
    ax.invert_yaxis()
    plt.tight_layout()
    plt.savefig(TOP_CORRELATIONS_IMAGE_PATH)
    plt.close()

if 'Exam_Score' in df.columns:
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.hist(df['Exam_Score'], bins=30, edgecolor='black', alpha=0.7)
    ax.set_xlabel('Exam Score')
    ax.set_ylabel('Frequency')
    ax.set_title('Distribution of Exam Scores')
    plt.tight_layout()
    plt.savefig(EXAM_SCORE_DISTRIBUTION_IMAGE_PATH)
    plt.close()
