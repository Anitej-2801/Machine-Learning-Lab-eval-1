import pandas as pd

# Load dataset
file_path = r"C:\Users\anite\Downloads\Lab Session Data(thyroid0387_UCI) (1).csv"
df = pd.read_csv(file_path)

# Automatically convert binary-like categorical columns to 0/1
binary_cols = []
for col in df.columns:
    unique_vals = df[col].dropna().unique()
    # Check for binary categorical like 'f'/'t' or 'yes'/'no'
    if set(unique_vals).issubset({'f', 't'}):
        df[col] = df[col].map({'f':0, 't':1})
        binary_cols.append(col)
    elif set(unique_vals).issubset({'no', 'yes'}):
        df[col] = df[col].map({'no':0, 'yes':1})
        binary_cols.append(col)
    elif set(unique_vals).issubset({0,1}):
        binary_cols.append(col)

if not binary_cols:
    print("No binary attributes found after conversion.")
else:
    print(f"Binary attributes considered: {binary_cols}")

    # Take first 2 observation vectors for binary attributes
    v1 = df.loc[0, binary_cols]
    v2 = df.loc[1, binary_cols]

    # Compute f11, f10, f01, f00
    f11 = sum((v1 == 1) & (v2 == 1))
    f10 = sum((v1 == 1) & (v2 == 0))
    f01 = sum((v1 == 0) & (v2 == 1))
    f00 = sum((v1 == 0) & (v2 == 0))

    print(f"\nf11 = {f11}, f10 = {f10}, f01 = {f01}, f00 = {f00}")

    # Calculate Jaccard Coefficient (JC) safely
    denominator_jc = f11 + f10 + f01
    JC = f11 / denominator_jc if denominator_jc != 0 else 0

    # Calculate Simple Matching Coefficient (SMC) safely
    denominator_smc = f11 + f10 + f01 + f00
    SMC = (f11 + f00) / denominator_smc if denominator_smc != 0 else 0

    print(f"Jaccard Coefficient (JC) = {JC:.3f}")
    print(f"Simple Matching Coefficient (SMC) = {SMC:.3f}")

    # Comparison insight
    print("\nComparison:")
    print("JC ignores attributes where both vectors have 0 (absence), making it suitable for presence-only features.")
    print("SMC considers both matches of 1 and 0, making it suitable when both presence and absence are meaningful.")

