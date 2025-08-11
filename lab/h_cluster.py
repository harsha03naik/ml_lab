import pandas as pd
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.cluster import AgglomerativeClustering

df = pd.read_csv(r"C:\Users\harsh\Desktop\mca\sem2\ML_py\lab\dataset\hierarchial.csv")

plt.scatter(df['x'], df['y'], marker='*', c='red', alpha=0.5)
plt.title('Raw Data Points')
plt.xlabel('X')
plt.ylabel('Y')
plt.show()

linked = linkage(df[['x', 'y']], method='single')
plt.figure(figsize=(8, 5))
dendrogram(linked, orientation='top', distance_sort='descending', show_leaf_counts=True)
plt.xlabel('Samples')
plt.ylabel('Euclidean Distances')
plt.title('Hierarchical Clustering Dendrogram')
plt.show()

model = AgglomerativeClustering(n_clusters=5, metric='euclidean', linkage='single')
model.fit(df[['x', 'y']])

plt.scatter(df['x'], df['y'], c=model.labels_, cmap='rainbow', s=50)
plt.xlabel('X')
plt.ylabel('Y')
plt.title('Hierarchical Clustering Result')
plt.grid(True)
plt.show()
