from sklearn.neighbors import LocalOutlierFactor
import numpy as np

X = np.array([1,2,2.5,3,3.5,10]).reshape(-1,1)
print(X)

lof = LocalOutlierFactor(n_neighbors=2)
y_pred = lof.fit_predict(X)  

print("Predictions:", y_pred)
for point, label in zip(X, y_pred):
    status = "Yes" if label == -1 else "No"
    print(f"Point {point} → Outlier? {status}")
