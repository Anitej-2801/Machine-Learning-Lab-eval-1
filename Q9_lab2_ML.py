import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler

# -----------------------------
# 1) Load dataset (use imputed version)
# -----------------------------
file_path = r"C:\Users\anite\Downloads\Purchase_data_imputed.csv"
df = pd.read_csv(file_path)

# -----------------------------
# 2) Separate numeric & categorical
# -----------------------------
num_cols = df.select_dtypes(include=[np.number]).columns
cat_cols = df.select_dtypes(exclude=[np.number]).columns

print("Numeric columns that may need normalization:")
print(list(num_cols))
print("-" * 50)

# -----------------------------
# 3) Choose scaling method
# -----------------------------
# Option 1: Min-Max Scaling (0–1 range)
minmax_scaler = MinMaxScaler()
df_minmax = df.copy()
df_minmax[num_cols] = minmax_scaler.fit_transform(df[num_cols])

# Option 2: Z-Score Standardization
standard_scaler = StandardScaler()
df_standard = df.copy()
df_standard[num_cols] = standard_scaler.fit_transform(df[num_cols])

# Option 3: Robust Scaling (handles outliers better)
robust_scaler = RobustScaler()
df_robust = df.copy()
df_robust[num_cols] = robust_scaler.fit_transform(df[num_cols])

# -----------------------------
# 4) Save normalized datasets
# -----------------------------
df_minmax.to_csv(r"C:\Users\anite\Downloads\Purchase_data_minmax.csv", index=False)
df_standard.to_csv(r"C:\Users\anite\Downloads\Purchase_data_standard.csv", index=False)
df_robust.to_csv(r"C:\Users\anite\Downloads\Purchase_data_robust.csv", index=False)

print("✅ Normalized datasets saved as:")
print(" - Purchase_data_minmax.csv")
print(" - Purchase_data_standard.csv")
print(" - Purchase_data_robust.csv")
 