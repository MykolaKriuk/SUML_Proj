import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder

df = pd.read_csv('../data/raw/StudentPerformanceFactors.csv')

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

# Label Encoding dla reszty
label_enc = LabelEncoder()
for col in cat_cols:
	df[col] = label_enc.fit_transform(df[col])

print(df.head())
print(df["Gender"])
print(df["Motivation_Level"])

df.to_csv("../data/cleaned/student_performance_cleaned.csv", index=False)
