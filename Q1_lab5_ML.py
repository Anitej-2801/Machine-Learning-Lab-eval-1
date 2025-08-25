import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

# Load the dataset
file_path = r"C:\Users\anite\Downloads\schizophrenia-features.csv (1).xlsx"
df = pd.read_excel(file_path)

# Select feature and target
X = df[['f.mean']]   # input feature (choose one numerical column)
y = df['class']      # target variable (numeric for regression)

# Split dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train Linear Regression model
reg = LinearRegression().fit(X_train, y_train)

# Predict on training set
y_train_pred = reg.predict(X_train)

# Print model coefficients
print("Coefficient (slope):", reg.coef_)
print("Intercept:", reg.intercept_)





