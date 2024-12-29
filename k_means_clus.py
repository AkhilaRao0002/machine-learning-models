#!/usr/bin/env python
# coding: utf-8

# In[2]:


pip install kneed


# In[ ]:


get_ipython().run_line_magic('matplotlib', 'inline')

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from kneed import KneeLocator


# In[ ]:


diabetes_dataset = pd.read_csv('diabetes.csv')
print(diabetes_dataset.head(25))
print(diabetes_dataset.describe())
f1=[]
for i in range(len(diabetes_dataset)):
    f1.append([diabetes_dataset.BMI[i],diabetes_dataset.Age[i]])


# In[ ]:


from sklearn.datasets import load_digits
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

pca = PCA(2)
df = pca.fit_transform(f1)
df.shape


# In[ ]:


sf=df
print(sf)


# In[ ]:


import warnings
from sklearn.cluster import KMeans

warnings.filterwarnings("ignore")

kmeans_kwargs = {
    "init": "random",
    "n_init": 10,
    "max_iter": 300,
    "random_state": 42,
}

ssel = []
for k in range(1, 11):
    kmeans = KMeans(n_clusters=k, **kmeans_kwargs)
    kmeans.fit(sf)
    ssel.append(kmeans.inertia_)

print(ssel)


# In[ ]:


plt.style.use("fivethirtyeight")
plt.plot(range(1, 11), ssel)
plt.xticks(range(1, 11))
plt.xlabel("Number of clusters")
plt.ylabel("SSE")
plt.show()


# In[ ]:


k1 = KneeLocator(
    range(1, 11),ssel, curve="convex", direction="decreasing")

k1.elbow


# In[ ]:


#shilouette_score=(separation-cohesion)/max(separation,cohesion)
from sklearn.metrics import silhouette_score
s_c= []

for k in range(2, 11):
    kmeans = KMeans(n_clusters=k, **kmeans_kwargs)
    kmeans.fit(sf)
    score=silhouette_score(sf, kmeans.labels_)
    s_c.append(score)
print(s_c)


# In[ ]:


plt.style.use("fivethirtyeight")
plt.plot(range(2, 11), s_c)
plt.xticks(range(2, 11))
plt.xlabel("Number of clusters")
plt.ylabel("Silh")
plt.show()


# In[ ]:


kmeans.fit(sf)


# In[ ]:


label = kmeans.predict(sf)


# In[ ]:


print(kmeans.n_iter_)
print(kmeans.inertia_)
kmeans.labels_


# In[ ]:


kmeans.cluster_centers_


# In[ ]:


filtered_label0=sf[label==0]
filtered_label1=sf[label==1]
filtered_label2=sf[label==2]
filtered_label3=sf[label==3]
plt.scatter(filtered_label0[:,0],filtered_label0[:,1],color="blue")
plt.scatter(filtered_label0[:,0],filtered_label0[:,1],color="red")
plt.scatter(filtered_label0[:,0],filtered_label0[:,1],color="black")
plt.scatter(filtered_label0[:,0],filtered_label0[:,1],color="yellow")


# In[4]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from kneed import KneeLocator
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import warnings

# Suppress warnings
warnings.filterwarnings("ignore")

# Load and display the dataset
diabetes_dataset = pd.read_csv('diabetes.csv')
print(diabetes_dataset.head(25))
print(diabetes_dataset.describe())

# Prepare the data for PCA
f1 = diabetes_dataset[['BMI', 'Age']].values

# Apply PCA to reduce dimensions to 2
pca = PCA(2)
df = pca.fit_transform(f1)
print(df)

# Determine the number of clusters using the elbow method
kmeans_kwargs = {
    "init": "random",
    "n_init": 10,
    "max_iter": 300,
    "random_state": 42,
}

sse = []
for k in range(1, 11):
    kmeans = KMeans(n_clusters=k, **kmeans_kwargs)
    kmeans.fit(df)
    sse.append(kmeans.inertia_)

# Plot SSE for the elbow method
plt.style.use("fivethirtyeight")
plt.plot(range(1, 11), sse)
plt.xticks(range(1, 11))
plt.xlabel("Number of clusters")
plt.ylabel("SSE")
plt.show()

# Find the optimal number of clusters using the knee locator
k1 = KneeLocator(range(1, 11), sse, curve="convex", direction="decreasing")
print(f"Optimal number of clusters: {k1.elbow}")

# Determine the number of clusters using the silhouette score
silhouette_coefficients = []
for k in range(2, 11):
    kmeans = KMeans(n_clusters=k, **kmeans_kwargs)
    kmeans.fit(df)
    score = silhouette_score(df, kmeans.labels_)
    silhouette_coefficients.append(score)

# Plot silhouette scores
plt.plot(range(2, 11), silhouette_coefficients)
plt.xticks(range(2, 11))
plt.xlabel("Number of clusters")
plt.ylabel("Silhouette Coefficient")
plt.show()

# Fit the final model with the optimal number of clusters
kmeans = KMeans(n_clusters=k1.elbow, **kmeans_kwargs)
kmeans.fit(df)
labels = kmeans.predict(df)

print(f"Number of iterations: {kmeans.n_iter_}")
print(f"Final SSE: {kmeans.inertia_}")
print("Cluster labels:", kmeans.labels_)
print("Cluster centers:", kmeans.cluster_centers_)

# Plot clusters
colors = ['blue', 'red', 'black', 'yellow']
for i in range(k1.elbow):
    filtered_label = df[labels == i]
    plt.scatter(filtered_label[:, 0], filtered_label[:, 1], color=colors[i])

plt.show()

