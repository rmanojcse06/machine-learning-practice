import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

X = np.array([[1], [2], [3], [4], [5], [6], [7], [8], [9], [10]])  # Feature
y = np.array([35000, 40000, 45000, 50000, 60000, 65000, 70000, 75000, 80000, 85000])  # Target

X,y = make_regression(
    n_samples=100,
    n_features=1,
    noise=10,
    random_state=42
)
print("X=",X)
print("y=",y)
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





import pandas as pd
import numpy as np
df = pd.read_csv('car_details_v4.csv')
df = df[(df["Make"] == "Honda") & (df["Fuel Type"] == "Petrol") & (df["Year"] > 1990) & (df["Model"].str.lower().str.startswith("city"))]


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test =train_test_split(X,y,test_size=0.20,random_state=40)

from sklearn.preprocessing import StandardScaler
standardization = StandardScaler()
X_train = standardization.fit_transform(X_train)
X_test = standardization.transform(X_test)


from sklearn.linear_model import LinearRegression
regression = LinearRegression()
regression.fit(X=X_train,y=y_train)

print("Coefficients (slope):", regression.coef_)
print("Intercept:", regression.intercept_)

y_pred=regression.predict(X_test)



import matplotlib.pyplot as plt
y_pred = regression.predict(X_test)

X_test_original = standardization.inverse_transform(X_test)
X_train_original = standardization.inverse_transform(X_train)

plt.scatter(X_train_original, y_train, color="blue", label="Training data")
#plt.plot(X_train, y, color="green", label="Test data")
plt.plot(X_test_original, y_pred, color="red", label="Prediction line")

plt.legend()
plt.show()
