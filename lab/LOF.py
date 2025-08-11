import numpy as np
from sklearn.neighbors import NearestNeighbors

data = np.array([1, 2, 2.5, 3, 3.5, 10]).reshape(-1, 1)
k = 2

nbrs = NearestNeighbors(n_neighbors=k+1)
nbrs.fit(data)
distances, indices = nbrs.kneighbors(data)

lrd = []
lof = []

for i in range(len(data)):
    reach_dists = []
    for j in range(1, k+1):  
        neighbor_idx = indices[i][j]
        neighbor_k_dist = distances[neighbor_idx][k]  
        actual_dist = distances[i][j]
        reach_dist = max(neighbor_k_dist, actual_dist)
        reach_dists.append(reach_dist)
    lrd_i = 1 / np.mean(reach_dists)
    lrd.append(lrd_i)

for i in range(len(data)):
    lrd_ratios = []
    for j in range(1, k+1):
        neighbor_idx = indices[i][j]
        lrd_ratios.append(lrd[neighbor_idx] / lrd[i])
    lof_i = np.mean(lrd_ratios)
    lof.append(lof_i)


print(f"{'Point':>6}|{'LRD':>8} |{'LOF':>8}| Outlier?")
print("-"*36)
for i in range(len(data)):
    is_outlier = "YES" if lof[i] > 1.5 else "NO"
    print(f"{data[i][0]:<5} | {lrd[i]:>6.2f} | {lof[i]:>7.2f} | {is_outlier}")
