# 3_id3_decision_tree.py
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

df = pd.read_csv(r"C:\Users\harsh\Desktop\mca\sem2\ML_py\lab\dataset\iris.csv")

column_names = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'class']
df.columns = column_names

print(df.isnull().sum())

X = df.drop('class', axis=1)
y = df['class']

print(f"Features shape: {X.shape}")
print(f"Target shape: {y.shape}")

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
print(f"Training set size: {X_train.shape[0]}")
print(f"Test set size: {X_test.shape[0]}")

model = DecisionTreeClassifier(criterion='entropy', random_state=42)
model.fit(X_train, y_train)

train_accuracy = model.score(X_train, y_train)
test_accuracy = model.score(X_test, y_test)
print(f"Train Accuracy: {train_accuracy:.4f}")
print(f"Test Accuracy: {test_accuracy:.4f}")

new_sample = [1.3, 1.3, 4.5, 8.0]
print(f"Sample length: {len(new_sample)}")

new_sample_df = pd.DataFrame([new_sample], columns=X.columns)

prediction = model.predict(new_sample_df)

print(f"Predicted Class: {prediction[0]}")
