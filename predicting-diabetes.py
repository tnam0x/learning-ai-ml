import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="numpy._core.getlimits")

import pandas as pd

from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

import matplotlib.pyplot as plt


# Load the diabetes dataset
data = load_diabetes()
X = pd.DataFrame(data.data, columns=data.feature_names)
y = pd.Series(data.target)

print("## Features ##")
print(X.sample(5))
print("\n## Describe ##")
print(X.describe())
print("\n## Target ##")
print(y.sample(5))

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = LinearRegression()
model.fit(X_train, y_train)

# Evaluate model
y_train_pred = model.predict(X_train)
y_test_pred = model.predict(X_test)
train_mse = mean_squared_error(y_train, y_train_pred)
test_mse = mean_squared_error(y_test, y_test_pred)

print("\n## Model Evaluation ##")
print(f"Training MSE: {train_mse:.2f}")
print(f"Testing MSE: {test_mse:.2f}")
print(f"Coefficients: {model.coef_.astype(int).tolist()}")
print(f"Intercept: {model.intercept_}")

# Plotting the results
plt.figure(figsize=(8, 6)) # Create a figure window, size 8x6 inches
plt.scatter(y_test, y_test_pred, color="purple", alpha=0.7) # Scatter plot of actual vs predicted
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=2) # Diagonal reference line
plt.xlabel("Actual Disease Progression")
plt.ylabel("Predicted Progression")
plt.title("Diabetes Test Prediction")
plt.grid(True) # Add grid for better readability
plt.show()

# Save as PNG
plt.savefig("diabetes_prediction.png")
print("Plot saved as diabetes_prediction.png")

# Save the model
import joblib
joblib.dump(model, "diabetes_model.pkl")
print("Model saved as diabetes_model.pkl")
