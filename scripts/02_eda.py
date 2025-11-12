import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Get paths relative to this script
SCRIPT_DIR = Path(__file__).parent
PROJECT_ROOT = SCRIPT_DIR.parent
DATA_CLEANED_PATH = PROJECT_ROOT / "data" / "cleaned" / "student_performance_cleaned.csv"
CORRELATION_IMAGE_PATH = PROJECT_ROOT / "data" / "images" / "correlation_matrix.png"

df = pd.read_csv(DATA_CLEANED_PATH)

corr = df.corr(numeric_only=True)

corr_unstacked = corr.abs().unstack().sort_values(ascending=False)
print("\nTop correlated pairs:")
print(corr_unstacked[(corr_unstacked < 1) & (corr_unstacked > 0.7)])

plt.figure(figsize=(16, 12))
sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f", square=True)
plt.title("Correlation Matrix of Student Performance Features")

# plt.tight_layout()
plt.savefig(CORRELATION_IMAGE_PATH)

plt.close()

threshold = 0.85
corr_matrix = df.corr(numeric_only=True).abs()
upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))

to_drop = [column for column in upper.columns if any(upper[column] > threshold)]
print("Highly correlated columns to drop:", to_drop)