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
