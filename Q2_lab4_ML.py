import pandas as pd
import numpy as np
from numpy.linalg import pinv, LinAlgError
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error, r2_score

# ===========================
# Q2: Model Evaluation (Fixed with Correct Columns)
# ===========================

# ✅ Your dataset file
file = r"C:\Users\anite\Downloads\Lab Session Data(Purchase data).csv"
df = pd.read_csv(file)

# --- Keep only the useful columns ---
df = df[["Customer", "Candies (#)", "Mangoes (Kg)", "Milk Packets (#)", "Payment (Rs)"]]

# --- Convert numeric columns ---
for col in df.columns[1:]:  # skip 'Customer'
    df[col] = pd.to_numeric(df[col], errors="coerce")

# Drop rows where Payment is missing
df = df.dropna(subset=["Payment (Rs)"])

# --- Split into A (features) and C (target payments) ---
A = df.iloc[:, 1:-1].values   # product quantities
C = df.iloc[:, -1].values     # actual payments

# --- Solve for product costs using pseudo-inverse (with fallback) ---
try:
    X_estimated = pinv(A).dot(C)
except LinAlgError:
    print("⚠️ SVD did not converge, falling back to least squares...")
    X_estimated, _, _, _ = np.linalg.lstsq(A, C, rcond=None)

# Predicted payments
C_pred = A.dot(X_estimated)

# --- Evaluation Metrics ---
mse = mean_squared_error(C, C_pred)
rmse = np.sqrt(mse)
mape = mean_absolute_percentage_error(C, C_pred)
r2 = r2_score(C, C_pred)

# --- Results ---
print("Mean Squared Error (MSE):", mse)
print("Root Mean Squared Error (RMSE):", rmse)
print("Mean Absolute Percentage Error (MAPE):", mape)
print("R2 Score:", r2)

# --- Actual vs Predicted Payments ---
comparison = pd.DataFrame({
    "Customer": df["Customer"],
    "Actual Payment": C,
    "Predicted Payment": np.round(C_pred, 2)
})
print("\n--- Actual vs Predicted Payments ---")
print(comparison.to_string(index=False))




