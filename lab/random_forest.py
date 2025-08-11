import numpy as np 
import pandas as pd 
from sklearn.ensemble import RandomForestClassifier 
from sklearn.cluster import AgglomerativeClustering 
x_real = np.array([ 
[1.0,2.0], 
[1.1,2.1], 
[8.0,9.0] 
]) 
n_samples, n_features = x_real.shape 
x_synthetic=np.zeros_like(x_real) 
for i in range(n_features): 
    x_synthetic[:,i]=np.random.permutation(x_real[:,i]) 
x_combined=np.vstack([x_real,x_synthetic]) 
y_labels=np.array([1]*n_samples+[0]*n_samples) 
rf=RandomForestClassifier(n_estimators=10,max_depth=3,random_state=42) 
rf.fit(x_combined,y_labels) 
leaf_indices=rf.apply(x_real) 
proximity=np.zeros((n_samples,n_samples)) 
for tree_leaf in leaf_indices.T: 
    for i in range(n_samples): 
        for j in range(n_samples): 
            if tree_leaf[i]==tree_leaf[j]: 
                proximity[i,j]+=1 
proximity/=rf.n_estimators 
distance=1-proximity 
clustering=AgglomerativeClustering(n_clusters=2,metric='precomputed',linkage='average') 
labels=clustering.fit_predict(distance) 
print("Real Data Points:") 
print(x_real) 
print("\nProximity Matrix:") 
print(np.round(proximity,2)) 
print("\nDistance Matrix:") 
print(np.round(distance,2)) 
print("\nCluster Labels:") 
print(labels)
