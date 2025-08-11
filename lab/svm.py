import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVC

X = np.array([
    (1, 0.5), (1, 1), (1, -0.5), (-0.5, 0.5), (0.5, 0.5), (2, 0),
    (4, 0), (4.5, 1), (5, -1), (5.5, 0)
])
y = np.array([-1, -1, -1, -1, -1, -1, 1, 1, 1, 1])

plt.figure(figsize=(8, 6))
plt.scatter(X[y == -1, 0], X[y == -1, 1], color='blue', label='Class -1')
plt.scatter(X[y == 1, 0], X[y == 1, 1], color='red', label='Class 1')

clf = SVC(kernel='linear', C=1.0)
clf.fit(X, y)

plt.scatter(
    clf.support_vectors_[:, 0], clf.support_vectors_[:, 1],
    s=120, facecolors='none', edgecolors='black', label='Support Vectors'
)

w = clf.coef_[0]
b = clf.intercept_[0]

if np.isclose(w[1], 0):
    x_hyperplane = -b / w[0]
    plt.axvline(x=x_hyperplane, color='green', linestyle='--', label='Hyperplane')
else:
    x_vals = np.linspace(min(X[:, 0]) - 1, max(X[:, 0]) + 1, 200)
    y_vals = -(w[0] * x_vals + b) / w[1]
    plt.plot(x_vals, y_vals, 'g--', label='Hyperplane')

plt.xlabel("X1")
plt.ylabel("X2")
plt.title("SVM Hyperplane")
plt.legend()
plt.grid(True)
plt.show()
