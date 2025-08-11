import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

df = pd.read_csv("C:\\Users\\harsh\\Desktop\mca\sem2\ML_py\\lab\\dataset\\kmeans.csv")

X = df[['Height', 'Weight']]

kmeans = KMeans(n_clusters=3, random_state=30)
df['cluster'] = kmeans.fit_predict(X)

print(df)

plt.scatter(df['Height'], df['Weight'], c=df['cluster'], cmap='rainbow', s=50)
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], c='black', marker='X', s=20, label='Centroids')
plt.xlabel('Height')
plt.ylabel('Weight')
plt.title('K-Means Clustering')
plt.legend()
plt.show()
