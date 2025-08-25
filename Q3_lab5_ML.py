import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

# Load the dataset
file_path = r"C:\Users\anite\Downloads\schizophrenia-features.csv (1).xlsx"
df = pd.read_excel(file_path)

# Remove any leading/trailing spaces from column names just in case
df.columns = df.columns.str.strip()

# Select multiple features (all numerical attributes)
X = df[['f.mean', 'f.sd', 'f.propZeros']]  # input features
y = df['class']                             # target variable

# Split dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train Linear Regression model
reg = LinearRegression().fit(X_train, y_train)

# Predict on training and test sets
y_train_pred = reg.predict(X_train)
y_test_pred = reg.predict(X_test)

# Define MAPE function
def mean_absolute_percentage_error(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    non_zero_idx = y_true != 0  # avoid division by zero
    return np.mean(np.abs((y_true[non_zero_idx] - y_pred[non_zero_idx]) / y_true[non_zero_idx])) * 100

# --- Metrics for training set ---
mse_train = mean_squared_error(y_train, y_train_pred)
rmse_train = np.sqrt(mse_train)
mape_train = mean_absolute_percentage_error(y_train, y_train_pred)
r2_train = r2_score(y_train, y_train_pred)

# --- Metrics for test set ---
mse_test = mean_squared_error(y_test, y_test_pred)
rmse_test = np.sqrt(mse_test)
mape_test = mean_absolute_percentage_error(y_test, y_test_pred)
r2_test = r2_score(y_test, y_test_pred)

# Print results
print("---- Training Set Metrics ----")
print("MSE:", mse_train)
print("RMSE:", rmse_train)
print("MAPE:", mape_train, "%")
print("R²:", r2_train)

print("\n---- Test Set Metrics ----")
print("MSE:", mse_test)
print("RMSE:", rmse_test)
print("MAPE:", mape_test, "%")
print("R²:", r2_test)

# Print model coefficients
print("\nLinear Regression Model Coefficients:")
for feature, coef in zip(X.columns, reg.coef_):
    print(f"{feature}: {coef}")
print("Intercept:", reg.intercept_)

