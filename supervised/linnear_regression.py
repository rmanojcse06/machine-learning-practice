import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

X = np.array([[1], [2], [3], [4], [5], [6], [7], [8], [9], [10]])  # Feature
y = np.array([35000, 40000, 45000, 50000, 60000, 65000, 70000, 75000, 80000, 85000])  # Target

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2, train_size=None, random_state=42)


model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

# Evaluate the model
print("Coefficients (slope):", model.coef_)
print("Intercept:", model.intercept_)
print("Mean squared error (MSE):", mean_squared_error(y_test, y_pred))
print("R² score:", r2_score(y_test, y_pred))



# Plotting
import matplotlib.pyplot as plt
plt.scatter(X, y, color='blue', label='Actual data')
plt.plot(X, model.predict(X), color='red', label='Regression line')
plt.xlabel("Years of Experience")
plt.ylabel("Salary")
plt.title("Linear Regression Example")
plt.legend()
plt.grid(True)
plt.show()