import pandas as pd
import numpy as np
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
import joblib

# Load the trained model
loaded_model = joblib.load("polynomial_regression_model.pkl")

# Example prediction with new data
height = 24
width = 3.5

# Define the polynomial degree used during training
degree = 2  # Example, adjust based on your model

# Create polynomial features for the new data
poly = PolynomialFeatures(degree=degree)
X_pred = poly.fit_transform(np.array([[height, width]]))

# Predict using the loaded model and transformed data
predicted_weight = loaded_model.predict(X_pred)

print(f"Predicted Weight: {predicted_weight[0]}")
