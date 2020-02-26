#!/usr/bin/env python
# coding: utf-8

# In[1]:


import matplotlib.pyplot as plt
from matplotlib import style
style.use('ggplot')
import numpy as np
from sklearn.datasets.samples_generator import make_blobs


# In[12]:


class Mean_Shift:
    
    # Constructor 
    def __init__(self, radius=3): # Default Radius is 3 (You can tune it accordingly)
        self.radius = radius
        self.centroids = {}
        
    # Create Clusters   
    def fit(self, data):
        centroids = {}

        # Your Code Here
        for i in range(len(data)):
            centroids[i] = data[i]
        while True:
            new_centroids = []
            for i in centroids:
                in_bandwidth = []
                centroid = centroids[i]
                for featureset in data:
                    if np.linalg.norm(featureset-centroid) < self.radius:
                        in_bandwidth.append(featureset)

                new_centroid = np.average(in_bandwidth,axis=0)
                new_centroids.append(tuple(new_centroid))

            uniques = sorted(list(set(new_centroids)))

            prev_centroids = dict(centroids)

            centroids = {}
            for i in range(len(uniques)):
                centroids[i] = np.array(uniques[i])

            optimized = True

            for i in centroids:
                if not np.array_equal(centroids[i], prev_centroids[i]):
                    optimized = False
                if not optimized:
                    break
                
            if optimized:
                break

        self.centroids = centroids
    
     


# In[26]:


# Creating Data Set of clusters
X, _ = make_blobs(n_samples = 200, cluster_std =1.5)


# In[27]:


# Creating Object of Class
clf = Mean_Shift()
clf.fit(X)

# Getting Optimized Centroids

centroids = clf.centroids

# Simple Scatter plot of 2D Data X
plt.scatter(X[:,0], X[:,1],s=150)

# Plot Cluster centroids as '*'
for c in centroids:
    plt.scatter(centroids[c][0], centroids[c][1], color='k', marker='*', s=150)

plt.show()


# In[ ]:




