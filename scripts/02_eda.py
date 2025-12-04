import pandas as pd
import numpy as np
from pathlib import Path

# Get paths relative to this script
SCRIPT_DIR = Path(__file__).parent
PROJECT_ROOT = SCRIPT_DIR.parent
DATA_CLEANED_PATH = PROJECT_ROOT / "data" / "cleaned" / "student_performance_cleaned.csv"

df = pd.read_csv(DATA_CLEANED_PATH)

# Correlation analysis
corr = df.corr(numeric_only=True)

corr_unstacked = corr.abs().unstack().sort_values(ascending=False)
print("\nTop correlated pairs:")
print(corr_unstacked[(corr_unstacked < 1) & (corr_unstacked > 0.7)])

# Highly correlated columns analysis
threshold = 0.85
corr_matrix = df.corr(numeric_only=True).abs()
upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))

to_drop = [column for column in upper.columns if any(upper[column] > threshold)]
print("Highly correlated columns to drop:", to_drop)