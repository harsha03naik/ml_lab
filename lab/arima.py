import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
import matplotlib.pyplot as plt

file_path = r"C:\Users\harsh\Desktop\mca\sem2\ML_py\lab\dataset\airline-passengers.csv"

df = pd.read_csv(file_path, parse_dates=['Month'], index_col='Month')

df.index.freq = 'MS'

model = ARIMA(df['Passengers'], order=(1, 2, 1))
model_fit = model.fit()

forecast = model_fit.forecast(steps=1).iloc[0]
print(f"Forecasted next value using ARIMA: {forecast:.2f}")

plt.figure(figsize=(10, 5))
plt.plot(df['Passengers'], label='Original Data')
plt.plot(model_fit.fittedvalues, label='Fitted Values', color='red')
plt.legend()
plt.title('ARIMA Forecasting on Airline Passengers')
plt.xlabel('Month')
plt.ylabel('Number of Passengers')
plt.grid(True)
plt.show()
