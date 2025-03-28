import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import KDTree
from sklearn.cluster import DBSCAN
import matplotlib
from mpl_toolkits.mplot3d import Axes3D
#%%
#Utility functions
def show_cloud(points_plt):
    ax = plt.axes(projection='3d')
    ax.scatter(points_plt[:,0], points_plt[:,1], points_plt[:,2], s=0.01)
    plt.show()

def show_scatter(x, y):
    plt.scatter(x, y)
    plt.show()
#%%
# Task 1:best value for ground level using histogram

def get_ground_level(pcd):
    hist, bin_edges = np.histogram(pcd[:, 2], bins=100)
    
    #bin with highest frequency
    grd_level = bin_edges[np.argmax(hist)]  
    return grd_level
#%%

# Loading files
pcd1 = np.load('D:/Lectures/Industrial AI/assignmnt 5/dataset1.npy')
pcd2 = np.load('D:/Lectures/Industrial AI/assignmnt 5/dataset2.npy')

# Task 1: grd level est for dataset 1
est_ground_level1 = get_ground_level(pcd1)
print(f"est ground level for dataset 1: {est_ground_level1}")

# points above the est grd level
pcd_above_ground1 = pcd1[pcd1[:, 2] > est_ground_level1]
show_cloud(pcd_above_ground1)

# Plot histogram
plt.hist(pcd1[:, 2], bins=100)
plt.title('histogram for dataset 1')
plt.xlabel('Z')
plt.ylabel('frequency')
plt.show()

#%%
# Task 1: grd level est for dataset 2
est_ground_level2 = get_ground_level(pcd2)
print(f"es Ground level for dataset 2: {est_ground_level2}")

# points above the est grd level
pcd_above_ground2 = pcd2[pcd2[:, 2] > est_ground_level2]
show_cloud(pcd_above_ground2)

# histogram of for Dataset 2
plt.hist(pcd2[:, 2], bins=100)
plt.title('histogram for dataset 2')
plt.xlabel('Z')
plt.ylabel('frequency')
plt.show()

#%%

# Task 2: optimal value for eps using elbow method

def find_optimal_eps(pcd_above_ground, k=10):

    #distance to the k-th neighbor
    tree = KDTree(pcd_above_ground[:, :2])
    distances, _ = tree.query(pcd_above_ground[:, :2], k+1)
    distances = np.sort(distances[:, -1])
    
    # Plot using elbow method
    plt.plot(distances)
    plt.title('optimal eps using elbow method')
    plt.xlabel('sorted points')
    plt.ylabel('distance to the nearest 10th neighbour')
    plt.grid(True)
    plt.show()

    # elbow point is the optimal eps value
    eps = distances[10]
    print(f"optimal eps got from elbow mrthod: {eps}")
    
    return eps

# optimal eps for dataset 1
optimal_eps1 = find_optimal_eps(pcd_above_ground1)

optimal_eps1 = 1.0
print(f"optimal eps for dataset 1: {optimal_eps1}")

# DBSCAN 
clustering_optimised1 = DBSCAN(eps=optimal_eps1, min_samples=5).fit(pcd_above_ground1)

# cluster result visualisation with the optimised eps for Dataset 1
clusters1 = len(set(clustering_optimised1.labels_)) - (1 if -1 in clustering_optimised1.labels_ else 0)
colors = [plt.cm.Spectral(each) for each in np.linspace(0, 1, clusters1)]
plt.figure(figsize=(10, 12))
plt.scatter(pcd_above_ground1[:, 0], pcd_above_ground1[:, 1], c=clustering_optimised1.labels_, cmap=matplotlib.colors.ListedColormap(colors), s=2)
plt.title(f'DBSCAN with optimised eps for Dataset 1: {optimal_eps1} ({clusters1} clusters)', fontsize=15)
plt.xlabel('X-axis', fontsize=10)
plt.ylabel('Y-axis', fontsize=10)
plt.show()

#%%

# ptimal eps for Dataset 2
optimal_eps2 = find_optimal_eps(pcd_above_ground2)

optimal_eps2 = 1.0
print(f"optimized eps for dataset 2: {optimal_eps2}")

# DBSCAN
clustering_optimised2 = DBSCAN(eps=optimal_eps2, min_samples=5).fit(pcd_above_ground2)

# Visualize the clustering result with the optimized eps for Dataset 2
clusters2 = len(set(clustering_optimised2.labels_)) - (1 if -1 in clustering_optimised2.labels_ else 0)
colors = [plt.cm.Spectral(each) for each in np.linspace(0, 1, clusters2)]
plt.figure(figsize=(10, 10))
plt.scatter(pcd_above_ground2[:, 0], pcd_above_ground2[:, 1], c=clustering_optimised2.labels_, cmap=matplotlib.colors.ListedColormap(colors), s=2)
plt.title(f'DBSCAN with optimized eps for Dataset 2: {optimal_eps2} ({clusters2} clusters)', fontsize=15)
plt.xlabel('X-axis', fontsize=10)
plt.ylabel('Y-axis', fontsize=10)
plt.show()

#%%
# Task 3: largest cluster for both datasets (catenary) 
def find_largest_cluster(pcd_above_ground, clustering_labels):
    clusters = np.unique(clustering_labels)
    largest_cluster = -1
    max_size = 0

    for cluster in clusters:
        if cluster != -1:  # Exclude noise cluster (-1)
            size = np.sum(clustering_labels == cluster)
            if size > max_size:
                max_size = size
                largest_cluster = cluster

    return largest_cluster, max_size

# cluster based on DBSCAN for dataset 1
largest_cluster1, size1 = find_largest_cluster(pcd_above_ground1, clustering_optimised1.labels_)

largest_cluster_points1 = pcd_above_ground1[clustering_optimised1.labels_ == largest_cluster1]

#largest cluster for Dataset 1
show_cloud(largest_cluster_points1)

print(f"largest cluster in dataset 1: {largest_cluster1}, Size: {size1}")

min_x1, min_y1 = np.min(largest_cluster_points1[:, 0]), np.min(largest_cluster_points1[:, 1])
max_x1, max_y1 = np.max(largest_cluster_points1[:, 0]), np.max(largest_cluster_points1[:, 1])
print(f"Dataset 1: Min(x): {min_x1}, Min(y): {min_y1}, Max(x): {max_x1}, Max(y): {max_y1}")

#%%

# cluster based on DBSCAN for dataset 2
largest_cluster2, size2 = find_largest_cluster(pcd_above_ground2, clustering_optimised2.labels_)

largest_cluster_points2 = pcd_above_ground2[clustering_optimised2.labels_ == largest_cluster2]

show_cloud(largest_cluster_points2)

print(f"largest cluster in dataset 2: {largest_cluster2}, Size: {size2}")

min_x2, min_y2 = np.min(largest_cluster_points2[:, 0]), np.min(largest_cluster_points2[:, 1])
max_x2, max_y2 = np.max(largest_cluster_points2[:, 0]), np.max(largest_cluster_points2[:, 1])
print(f"dataset 2: Min(x): {min_x2}, Min(y): {min_y2}, Max(x): {max_x2}, Max(y): {max_y2}")

noise_points1 = np.sum(clustering_optimised1.labels_ == -1)
noise_points2 = np.sum(clustering_optimised2.labels_ == -1)

print(f"no of noise points in dataset 1: {noise_points1}")
print(f"no of noise points in dataset 1: {noise_points2}")
