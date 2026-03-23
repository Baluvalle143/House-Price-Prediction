import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
import matplotlib.pyplot as plt


df = pd.read_csv("data.csv")


X = df[['area', 'bedrooms']]
y = df['price']


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)


model = LinearRegression()
model.fit(X_train, y_train)


predictions = model.predict(X_test)


error = mean_absolute_error(y_test, predictions)
print("Mean Absolute Error:", error)


new_house = [[2200, 3]]
predicted_price = model.predict(new_house)
print("Predicted Price:", predicted_price[0])

# Graph visualization
plt.scatter(df['area'], df['price'])
plt.xlabel("Area")
plt.ylabel("Price")
plt.title("Area vs Price")
plt.show()
