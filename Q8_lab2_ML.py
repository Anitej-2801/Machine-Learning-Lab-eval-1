import pandas as pd
import numpy as np

# -----------------------------
# 1) Load dataset
# -----------------------------
file_path = r"C:\Users\anite\Downloads\Lab Session Data(Purchase data).csv"
df = pd.read_csv(file_path)

print("Before imputation: Missing values count per column")
print(df.isnull().sum())
print("-" * 50)

# -----------------------------
# 2) Separate numeric & categorical
# -----------------------------
num_cols = df.select_dtypes(include=[np.number]).columns
cat_cols = df.select_dtypes(exclude=[np.number]).columns

# -----------------------------
# 3) Function to check outliers (IQR method)
# -----------------------------
def has_outliers(series):
    q1 = series.quantile(0.25)
    q3 = series.quantile(0.75)
    iqr = q3 - q1
    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr
    return ((series < lower) | (series > upper)).any()

# -----------------------------
# 4) Imputation process
# -----------------------------
for col in num_cols:
    if df[col].isnull().sum() > 0:   # only if missing
        if has_outliers(df[col].dropna()):
            fill_value = df[col].median()
            df[col].fillna(fill_value, inplace=True)
            print(f"{col}: filled missing with Median = {fill_value}")
        else:
            fill_value = df[col].mean()
            df[col].fillna(fill_value, inplace=True)
            print(f"{col}: filled missing with Mean = {fill_value}")

for col in cat_cols:
    if df[col].isnull().sum() > 0:   # only if missing
        fill_value = df[col].mode()[0]
        df[col].fillna(fill_value, inplace=True)
        print(f"{col}: filled missing with Mode = {fill_value}")

# -----------------------------
# 5) Confirm no missing values remain
# -----------------------------
print("\nAfter imputation: Missing values count per column")
print(df.isnull().sum())

# -----------------------------
# 6) (Optional) Save cleaned dataset
# -----------------------------
output_file = r"C:\Users\anite\Downloads\Purchase_data_imputed.csv"
df.to_csv(output_file, index=False)
print(f"\nImputed dataset saved as: {output_file}")
