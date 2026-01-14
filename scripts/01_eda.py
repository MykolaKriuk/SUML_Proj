import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from pathlib import Path

# Get paths relative to this script
SCRIPT_DIR = Path(__file__).parent
PROJECT_ROOT = SCRIPT_DIR.parent
DATA_RAW_PATH = PROJECT_ROOT / "data" / "raw" / "StudentPerformanceFactors.csv"
DATA_CLEANED_PATH = PROJECT_ROOT / "data" / "cleaned" / "student_performance_cleaned.csv"

df = pd.read_csv(DATA_RAW_PATH)

df.head()  # pierwszy rzut oka
df.info()  # typy danych, liczba nie-null
df.describe()  # statystyki numeryczne
df.columns  # lista kolumn
# print(df.isna().sum())
print(df.isnull().sum())

num_cols = df.select_dtypes(include=['int64', 'float64']).columns
cat_cols = df.select_dtypes(include=['object', 'category']).columns

# Uzupełnianie braków
for col in ["Parental_Education_Level", "Teacher_Quality", "Distance_from_Home"]:
	df[col] = df[col].fillna("Unknown")

print()
print("Po dodaniu unknown:")
print(df.isnull().sum())

# Tworzenie nowych cech
df["study_efficiency"] = df["Hours_Studied"] / df["Sleep_Hours"].replace(0, 0.5)
df["effective_study_hours"] = df["Hours_Studied"] + 2 * df["Tutoring_Sessions"]

# Mapowanie binarne
binary_maps = {
	"Gender": {"Male": 0, "Female": 1},
	"Internet_Access": {"No": 0, "Yes": 1},
	"Extracurricular_Activities": {"No": 0, "Yes": 1},
	"Learning_Disabilities": {"No": 0, "Yes": 1},
}
for col, mapping in binary_maps.items():
	df[col] = df[col].map(mapping)

# Label Encoding dla reszty (wykluczamy kolumny już przetworzone przez mapowanie binarne)
# Exclude columns that were already binary encoded
binary_cols = set(binary_maps.keys())
remaining_cat_cols = [col for col in cat_cols if col not in binary_cols]

# Create a new LabelEncoder for each column to avoid state overwriting
for col in remaining_cat_cols:
	label_enc = LabelEncoder()
	df[col] = label_enc.fit_transform(df[col])

print(df.head())
print(df["Gender"])
print(df["Motivation_Level"])

df.to_csv(DATA_CLEANED_PATH, index=False)

# Graphics generation
IMAGES_DIR = PROJECT_ROOT / "data" / "images"
CORRELATION_IMAGE_PATH = IMAGES_DIR / "correlation_matrix.png"
TOP_CORRELATIONS_IMAGE_PATH = IMAGES_DIR / "top_correlations.png"
EXAM_SCORE_DISTRIBUTION_IMAGE_PATH = IMAGES_DIR / "exam_score_distribution.png"

# Ensure images directory exists
IMAGES_DIR.mkdir(parents=True, exist_ok=True)

# Correlation Matrix
corr = df.corr(numeric_only=True)

corr_unstacked = corr.abs().unstack().sort_values(ascending=False)
print("\nTop correlated pairs:")
print(corr_unstacked[(corr_unstacked < 1) & (corr_unstacked > 0.7)])

plt.figure(figsize=(16, 12))
sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f", square=True)
plt.title("Correlation Matrix of Student Performance Features")
plt.savefig(CORRELATION_IMAGE_PATH)
plt.close()

# Highly correlated columns analysis
threshold = 0.85
corr_matrix = df.corr(numeric_only=True).abs()
upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))

to_drop = [column for column in upper.columns if any(upper[column] > threshold)]
print("\nHighly correlated columns to drop (threshold > 0.85):", to_drop)

# Top Correlations with Exam Score
if 'Exam_Score' in df.columns:
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    correlations = df[numeric_cols].corr()['Exam_Score'].abs().sort_values(ascending=False)
    correlations = correlations[correlations.index != 'Exam_Score']
    
    # Visualization of top correlations
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

# Exam Score Distribution
if 'Exam_Score' in df.columns:
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.hist(df['Exam_Score'], bins=30, edgecolor='black', alpha=0.7)
    ax.set_xlabel('Exam Score')
    ax.set_ylabel('Frequency')
    ax.set_title('Distribution of Exam Scores')
    plt.tight_layout()
    plt.savefig(EXAM_SCORE_DISTRIBUTION_IMAGE_PATH)
    plt.close()
