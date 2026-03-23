import pandas as pd
from sklearn.linear_model import LinearRegression


data = {
    'area': [1000, 1500, 2000, 2500, 3000],
    'bedrooms': [2, 3, 3, 4, 4],
    'price': [300000, 400000, 500000, 600000, 700000]
}

df = pd.DataFrame(data)


X = df[['area', 'bedrooms']]
y = df['price']


model = LinearRegression()
model.fit(X, y)


new_house = [[2200, 3]]
predicted_price = model.predict(new_house)

print("Predicted Price:", predicted_price[0])