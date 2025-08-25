import pandas as pd
import numpy as np

# -----------------------------
# 1) Load data
# -----------------------------
file_path = r"C:\Users\anite\Downloads\Lab Session Data(thyroid0387_UCI) (1).csv"
df = pd.read_csv(file_path)

# -----------------------------
# 2) Normalize binary-like values
# -----------------------------
def to_lower_str(x):
    return x.lower() if isinstance(x, str) else x

df = df.applymap(to_lower_str)

binary_maps = [
    ({'t','f'}, {'t':1, 'f':0}),
    ({'yes','no'}, {'yes':1, 'no':0}),
    ({'true','false'}, {'true':1, 'false':0})
]

for col in df.columns:
    vals = set(df[col].dropna().unique())
    for keyset, mapping in binary_maps:
        if vals and vals.issubset(keyset):
            df[col] = df[col].map(mapping)
            break

# -----------------------------
# 3) One-hot encode ALL non-numeric columns
# -----------------------------
df_encoded = pd.get_dummies(df, drop_first=False, dummy_na=False)

# -----------------------------
# 4) Handle missing values (fill NaNs with 0)
# -----------------------------
df_encoded = df_encoded.fillna(0)

# -----------------------------
# 5) Extract first two rows as vectors
# -----------------------------
a = df_encoded.iloc[0].to_numpy(dtype=float)
b = df_encoded.iloc[1].to_numpy(dtype=float)

# -----------------------------
# 6) Compute cosine similarity
# -----------------------------
dot_ab = float(np.dot(a, b))
norm_a = float(np.linalg.norm(a))
norm_b = float(np.linalg.norm(b))

if norm_a == 0 or norm_b == 0:
    cos_sim = 0.0
else:
    cos_sim = dot_ab / (norm_a * norm_b)

print(f"Number of features in complete vector: {df_encoded.shape[1]}")
print(f"Dot product <A,B> = {dot_ab:.6f}")
print(f"||A|| = {norm_a:.6f}, ||B|| = {norm_b:.6f}")
print(f"Cosine Similarity = {cos_sim:.6f}")

