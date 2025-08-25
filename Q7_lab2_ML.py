import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# -----------------------------
# 1) Load data
# -----------------------------
file_path = r"C:\Users\anite\Downloads\Lab Session Data(thyroid0387_UCI) (1).csv"
df = pd.read_csv(file_path)

# -----------------------------
# 2) Preprocess: binary conversion (t/f, yes/no → 0/1)
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
# 3) Select first 20 observations
# -----------------------------
df20 = df.head(20)

# -----------------------------
# 4) Identify binary attributes
# -----------------------------
binary_cols = [col for col in df20.columns if set(df20[col].dropna().unique()).issubset({0,1})]

# -----------------------------
# 5) Jaccard & SMC (binary only)
# -----------------------------
def jaccard_smc(v1, v2):
    f11 = np.sum((v1==1) & (v2==1))
    f10 = np.sum((v1==1) & (v2==0))
    f01 = np.sum((v1==0) & (v2==1))
    f00 = np.sum((v1==0) & (v2==0))
    jc = f11 / (f11+f10+f01) if (f11+f10+f01) != 0 else 0
    smc = (f11+f00) / (f11+f10+f01+f00) if (f11+f10+f01+f00) != 0 else 0
    return jc, smc

n = len(df20)
jc_matrix = np.zeros((n,n))
smc_matrix = np.zeros((n,n))

for i in range(n):
    for j in range(n):
        jc, smc = jaccard_smc(df20.iloc[i][binary_cols], df20.iloc[j][binary_cols])
        jc_matrix[i,j] = jc
        smc_matrix[i,j] = smc

# -----------------------------
# 6) Cosine Similarity (all attributes, one-hot encode categoricals)
# -----------------------------
df_encoded = pd.get_dummies(df20, drop_first=False, dummy_na=False)
df_encoded = df_encoded.fillna(0)

X = df_encoded.to_numpy(dtype=float)

cos_matrix = np.zeros((n,n))
for i in range(n):
    for j in range(n):
        dot = np.dot(X[i], X[j])
        norm_i = np.linalg.norm(X[i])
        norm_j = np.linalg.norm(X[j])
        cos_matrix[i,j] = dot / (norm_i*norm_j) if norm_i!=0 and norm_j!=0 else 0

# -----------------------------
# 7) Convert to DataFrames for plotting
# -----------------------------
labels = [f"Obs{i+1}" for i in range(n)]
jc_df = pd.DataFrame(jc_matrix, index=labels, columns=labels)
smc_df = pd.DataFrame(smc_matrix, index=labels, columns=labels)
cos_df = pd.DataFrame(cos_matrix, index=labels, columns=labels)

# -----------------------------
# 8) Plot heatmaps
# -----------------------------
plt.figure(figsize=(18,5))

plt.subplot(1,3,1)
sns.heatmap(jc_df, annot=False, cmap="viridis")
plt.title("Jaccard Similarity (Binary)")

plt.subplot(1,3,2)
sns.heatmap(smc_df, annot=False, cmap="viridis")
plt.title("Simple Matching Coefficient (Binary)")

plt.subplot(1,3,3)
sns.heatmap(cos_df, annot=False, cmap="viridis")
plt.title("Cosine Similarity (All Features)")

plt.tight_layout()
plt.show()
