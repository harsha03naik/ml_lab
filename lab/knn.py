import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.neighbors import KNeighborsClassifier

df = pd.read_csv("C:\\Users\\harsh\\Desktop\\mca\\sem2\\ML_py\\lab\\dataset\\IRIS.csv")

X = df.drop("class", axis=1).values
y = df["class"].values

le = LabelEncoder()
y_encoded = le.fit_transform(y)

scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y_encoded, test_size=0.5, random_state=42
)

knn = KNeighborsClassifier(n_neighbors=10)
knn.fit(X_train, y_train)

y_pred = knn.predict(X_test)

print("\nCorrect Predictions:")
for actual, predicted in zip(y_test, y_pred):
    if actual == predicted:
        print(f"Actual: {le.inverse_transform([actual])[0]}, Predicted: {le.inverse_transform([predicted])[0]}")

print("\nWrong Predictions:")
for actual, predicted in zip(y_test, y_pred):
    if actual != predicted:
        print(f"Actual: {le.inverse_transform([actual])[0]}, Predicted: {le.inverse_transform([predicted])[0]}")
