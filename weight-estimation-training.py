import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import joblib

# Read data from CSV file
df = pd.read_csv('test2.csv')

# Features (independent variables)
X = df[['Height', 'Width']]
# Target (dependent variable)
y = df['Weight']

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create polynomial features
degree = 2  # You can change the degree of the polynomial
poly = PolynomialFeatures(degree)
X_train_poly = poly.fit_transform(X_train)
X_test_poly = poly.transform(X_test)

# Create linear regression object
model = LinearRegression()

# Train the model using the training sets with polynomial features
model.fit(X_train_poly, y_train)

# Make predictions using the testing set
y_pred = model.predict(X_test_poly)

# Save the model to a file
joblib.dump(model, 'polynomial_regression_model.pkl')

# Print coefficients and intercept
print('Coefficients:', model.coef_)
print('Intercept:', model.intercept_)

# Print performance metrics
print('Mean squared error (MSE):', mean_squared_error(y_test, y_pred))
print('Coefficient of determination (R^2):', r2_score(y_test, y_pred))

# Compare actual vs predicted
comparison_df = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})
print(comparison_df)

# Plot outputs: Actual vs Predicted
plt.figure(figsize=(15, 5))

# Scatter plot for actual vs predicted weights
plt.subplot(1, 3, 1)
plt.scatter(y_test, y_pred)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], color='red', lw=2)  # Line for perfect prediction
plt.xlabel('Actual Weight')
plt.ylabel('Predicted Weight')
plt.title('Actual vs Predicted Weight')

# Scatter plot for Height vs Weight with regression line
plt.subplot(1, 3, 2)
plt.scatter(X_test['Height'], y_test, color='blue', label='Actual')
plt.scatter(X_test['Height'], y_pred, color='orange', label='Predicted')
plt.plot(X_test['Height'], model.predict(poly.transform(X_test)), color='red', lw=2, label='Regression Line')
plt.xlabel('Height')
plt.ylabel('Weight')
plt.title('Height vs Weight')
plt.legend()

# Scatter plot for Width vs Weight with regression line
plt.subplot(1, 3, 3)
plt.scatter(X_test['Width'], y_test, color='blue', label='Actual')
plt.scatter(X_test['Width'], y_pred, color='orange', label='Predicted')
plt.plot(X_test['Width'], model.predict(poly.transform(X_test)), color='red', lw=2, label='Regression Line')
plt.xlabel('Width')
plt.ylabel('Weight')
plt.title('Width vs Weight')
plt.legend()

# Adjust layout and show plot
plt.tight_layout()
plt.show()
