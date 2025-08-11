# 1_linear_regression.py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model

# Load dataset
df = pd.read_csv("C:\\Users\\harsh\\Desktop\\mca\\sem2\\ML_py\\lab\\dataset\\pizza.csv")

# Plot scatter
plt.xlabel('Size')
plt.ylabel('Price')
plt.scatter(df['size'], df['price'], color='red', marker='+')

# Create and train model
reg = linear_model.LinearRegression()
reg.fit(df[['size']], df['price'])

# Plot regression line
plt.plot(df['size'], reg.predict(df[['size']]), color='blue')
plt.title("Pizza Size vs Price")
plt.show()

# Predict for size 15
predicted_price = reg.predict(np.array([[15]]))
print("Predicted price for size 15 pizza:", predicted_price[0])
