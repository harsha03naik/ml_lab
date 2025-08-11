# 4_naive_bayes.py
import pandas as pd
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

data = pd.read_csv("C:\\Users\\harsh\\Desktop\\mca\\sem2\\ML_py\\lab\\dataset\\IRIS.csv")

x = data.drop(["class"], axis=1)
y = data["class"]

scaler = MinMaxScaler()
x_scaled = scaler.fit_transform(x)

X_train, X_test, y_train, y_test = train_test_split(x_scaled, y, test_size=0.2, random_state=42)

gnb = GaussianNB()
gnb.fit(X_train, y_train)

y_pred = gnb.predict(X_test)
print(y_pred)
