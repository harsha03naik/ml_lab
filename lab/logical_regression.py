import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

dataset = pd.read_csv("C:\\Users\\harsh\\Desktop\\mca\\sem2\\ML_py\\lab\\dataset\\insurance.csv")

X = dataset[['age']]
y = dataset['insurance']

X_train,X_test,y_train, y_test = train_test_split(X, y, train_size=0.2, random_state=1)

lg = LogisticRegression()

lg.fit(X_train,y_train)
predict = lg.predict(X_test)

accuracy = accuracy_score(y_test, predict)
print(round(accuracy,2))

cm = confusion_matrix(y_test, predict)
print("Confusion Matrix:\n", cm)

print("Classification Report:\n", classification_report(y_test, predict))

x_sorted = X.sort_values(by='age')
y_prob = lg.predict_proba(x_sorted)[:, 1]

plt.scatter(X, y, color='red', label='Data Points')
plt.plot(x_sorted, y_prob, color='blue', label='Sigmoid Curve')

plt.xlabel('Age')
plt.ylabel('Probability of Buying Insurance')
plt.legend()
plt.show()
