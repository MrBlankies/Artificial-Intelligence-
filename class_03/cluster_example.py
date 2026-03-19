import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from mpl_toolkits.mplot3d import Axes3D
from sklearn.preprocessing import LabelEncoder
import plotly.express as px


### load the dataset
df = pd.read_csv("Mall_Customers.csv")
#print(df.head())

### quick overview of the discriptive analytics of the dataset
#print(df.info())

#print(df.describe())

### Select features for clustering
### we need to define X matrix

X = df[['Annual Income (k$)', 'Spending Score (1-100)']] # 2-dimension clustering

kmeans = KMeans(n_clusters=5, random_state=42) # calculate the clusters

### fit the model

df['Cluster'] = kmeans.fit_predict(X) # create a new column with the "predictions"

#print(df.head())

### Plot the clusters
plt.figure(figsize=(10,5))
plt.scatter(
    df['Annual Income (k$)'],
    df['Spending Score (1-100)'],
    c=df['Cluster']
)

### Plot the centroids
centroids = kmeans.cluster_centers_
plt.scatter(
    centroids[:,0],
    centroids[:,1],
    marker='X',
    s=100
)

plt.title("Customer segmentation using k-means")
plt.xlabel("Annual Income (k$)")
plt.ylabel("Spending Score (1-100)")
plt.show()

### a 3-d cluster
### we are going to add one more feature

X = df[['Annual Income (k$)', 'Spending Score (1-100)', 'Age']]

kmeans = KMeans(n_clusters=5, random_state=42)
df['Cluster'] = kmeans.fit_predict(X)

#print(df.head())

### Plot the 3d cluster

fig = plt.figure(figsize=(12,8))
ax = fig.add_subplot(111, projection='3d')

### Plot clusters
ax.scatter(
    df['Annual Income (k$)'],
    df['Spending Score (1-100)'],
    df['Age'],
    c=df['Cluster'],
    cmap='viridis',
    s=60
)

### Plot centroids
centroids = kmeans.cluster_centers_
ax.scatter(
    centroids[:,0],
    centroids[:,1],
    centroids[:,2],
    marker='X',
    s=100,
    c='red',
    label = 'Centroids'
)

ax.set_title("3D Customer segmentation using k-means")
ax.set_xlabel('Annual Income (k$)')
ax.set_ylabel('Spending Score (1-100)')
ax.set_zlabel('Age')

plt.legend()
plt.show()

### 2D plot annual income vs age

X = df[['Spending Score (1-100)','Age']]

kmeans = KMeans(n_clusters=5,random_state=42)
df['Cluster']=kmeans.fit_predict(X)

plt.figure(figsize=(12,8))
plt.scatter(
    df['Spending Score (1-100)'],
    df['Age'],
    c=df['Cluster']
)
centroids = kmeans.cluster_centers_
plt.scatter(
    centroids[:,0],
    centroids[:,1],
    marker='X',
    s=100
)

plt.title("Customer segmentation using k-means")
plt.xlabel("Spending Score (1-100)")
plt.ylabel("Age")

plt.show()

### 2D plot annual income vs age

X = df[['Annual Income (k$)','Age']]

kmeans = KMeans(n_clusters=5,random_state=42)
df['Cluster']=kmeans.fit_predict(X)

plt.figure(figsize=(12,8))
plt.scatter(
    df['Annual Income (k$)'],
    df['Age'],
    c=df['Cluster']
)
centroids = kmeans.cluster_centers_
plt.scatter(
    centroids[:,0],
    centroids[:,1],
    marker='X',
    s=100
)

plt.title("Customer segmentation using k-means")
plt.xlabel("Annual Income (k$)")
plt.ylabel("Age")

plt.show()

### gender is Male/Female, it is categorical, for kmeans we need to tranform categorical data to numerical data

### encode Gender (Male=0, Female=1)

le =LabelEncoder()
df['Gender_encoded'] = le.fit_transform(df['Genre'])
print(df.head())

### Plot the 4d cluster

X = df[['Annual Income (k$)', 'Spending Score (1-100)', 'Age', 'Gender_encoded']]

kmeans = KMeans(n_clusters=5, random_state=42)
df['Cluster4d']=kmeans.fit_predict(X)

# Plot clusters
fig = px.scatter_3d(
    df,
    x='Annual Income (k$)',
    y='Spending Score (1-100)',
    z='Age',
    color='Cluster',
    symbol='Gender_encoded',
    title = "4D segmentation"
)

#Plot centroids
fig.show()