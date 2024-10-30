import numpy as np
from scipy.spatial.distance import pdist, squareform
import matplotlib.pyplot as plt
import pandas as pd
from copy import deepcopy

class my_HierarchicalClustering:
    def __init__(self,n_clusters,method='min'):
        self.n_clusters = n_clusters
        self.method = method

    def lance_williams(self, r_us, r_vs, r_uv, u_len, v_len, s_len, method='min'):
        """Apply the Lance-Williams formula to compute the new distance between merged clusters."""
        au = av = b = g = 0

        if method == 'min':  # Single linkage (minimum distance)
            au = 0.5
            av = 0.5
            b = 0
            g = -0.5
        elif method == 'max':  # Complete linkage (maximum distance)
            au = 0.5
            av = 0.5
            b = 0
            g = 0.5
        elif method == 'mean':  # Average linkage (mean distance)
            au = u_len / (u_len + v_len)
            av = v_len / (u_len + v_len)
            b = g = 0
        elif method == 'center':  # Centroid linkage
            au = u_len / (u_len + v_len)
            av = v_len / (u_len + v_len)
            b = -au * av
            g = 0
        elif method == 'ward':  # Ward's method
            au = (s_len + u_len) / (s_len + u_len + v_len)
            av = (s_len + v_len) / (s_len + u_len + v_len)
            b = -s_len / (s_len + u_len + v_len)
            g = 0
        else:
            raise Exception("Invalid method. Choose one of: 'min', 'max', 'mean', 'center', 'ward'.")

        return au * r_us + av * r_vs + b * r_uv + g * np.abs(r_us - r_vs)

    def init_clusters(self, n):
        clusters = [[i] for i in range(n)]
        cluster_flags = [True for _ in range(n)]

        return clusters,cluster_flags
    
    def calculate_distance_matrix(self,data,n):
        distance_matrix = np.zeros((n,n))
        
        for i in range(n):
            for j in range(i+1,n):
                dist = np.linalg.norm(data.iloc[i] - data.iloc[j])
                distance_matrix[i][j] = dist
                distance_matrix[j][i] = dist

        return distance_matrix

    def find_closest_clusters(self,clusters,cluster_flags,distance_matrix):
        min_dist = np.inf
        pair = (0,0)

        for i in range(len(clusters)):
            for j in range(i+1,len(clusters)):
                if cluster_flags[i] and cluster_flags[j]:
                    if distance_matrix[i][j] < min_dist:
                        min_dist = distance_matrix[i][j]
                        pair = (i,j)
        return pair[0], pair[1], min_dist
    
    def merge_clusters(self,clusters,c1,c2):
        return clusters[c1] + clusters[c2]
    
    def update_distance_matrix(self,distance_matrix,clusters,cluster_flags,c1,c2):
        distance_matrix = np.append(distance_matrix,np.zeros((len(clusters) - 1,1)), axis=1)
        distance_matrix = np.append(distance_matrix,np.zeros((1,len(clusters))), axis= 0)

        for i in range(len(clusters) - 1):
            if i != c1 and i != c2:
                if cluster_flags[i]:
                    r_us = distance_matrix[c1][i]
                    r_vs = distance_matrix[c2][i]
                    r_uv = distance_matrix[c1][c2]
                    u_len = len(clusters[c1])
                    v_len = len(clusters[c2])
                    s_len = len(clusters[i])

                    new_d = self.lance_williams(r_us,r_vs,r_uv,u_len,v_len,s_len,method=self.method)
                    distance_matrix[i][-1] = new_d
                    distance_matrix[-1][i] = new_d
    
        return distance_matrix
    
    def perform_clustering(self, distance_matrix, clusters,cluster_flags):
        linkage_matrix = []
        linkage_history = []


        while cluster_flags.count(True) > self.n_clusters:
            c1,c2, min_dist = self.find_closest_clusters(clusters,cluster_flags,distance_matrix)

            newc = self.merge_clusters(clusters,c1,c2)

            linkage_matrix.append([c1,c2,min_dist, len(newc)])

            cluster_flags[c1] = False
            cluster_flags[c2] = False
            cluster_flags.append(True)
            linkage_history.append(deepcopy(cluster_flags))
            clusters.append(newc)
            
            distance_matrix = self.update_distance_matrix(distance_matrix,clusters,cluster_flags,c1,c2)


        return linkage_history,linkage_matrix


    
    def fit(self,data):
        n = data.shape[0]

        clusters ,cluster_flags = self.init_clusters(n)

        distance_matrix = self.calculate_distance_matrix(data,n)

        linkage_history , linkage_matrix = self.perform_clustering(distance_matrix,clusters,cluster_flags)

        self.labels_ = []
        for i in range(len(clusters)):
            if cluster_flags[i]:
                c1 = []
                for j in range(len(clusters[i])):
                    c1.append(data.iloc[clusters[i][j]].to_numpy().tolist())
                self.labels_.append(c1)

        return linkage_history, linkage_matrix
        
    def plot_clusters(self):
        """Plot the clusters."""
        if self.labels_ is None:
            raise ValueError("You must fit the model before plotting.")
        
        plt.figure(figsize=(8, 6))
        
        for i in range(len(self.labels_)):
            plt.scatter(np.array(self.labels_[i])[:, 0], np.array(self.labels_[i])[:, 1])
        
        plt.title(f'Hierarchical Clustering (n_clusters={self.n_clusters}, linkage={self.method})')
        plt.xlabel('Feature 1')
        plt.ylabel('Feature 2')
        plt.grid(True)
        plt.show()


def plot_all_linkage_methods(X, n_clusters=2):
    """Plot hierarchical clustering with different linkage methods using unique colors for each cluster."""
    linkage_methods = ['min', 'max', 'mean', 'center', 'ward']
    fig, axs = plt.subplots(1, len(linkage_methods), figsize=(20, 5))
    
    for i, method in enumerate(linkage_methods):
        model = my_HierarchicalClustering(n_clusters=n_clusters, method=method)
        model.fit(X)
        
        for j in range(len(model.labels_)):
            axs[i].scatter(np.array(model.labels_[j])[:, 0], np.array(model.labels_[j])[:, 1])

        axs[i].set_title(f'Linkage: {method}')
        axs[i].grid(True)

    plt.tight_layout()
    plt.show()


# class my_HierarchicalClustering:
#     def __init__(self, n_clusters=2, linkage='min'):
#         self.n_clusters = n_clusters
#         self.linkage = linkage
#         self.labels_ = None
#         self.linkage_matrix = []

#     def lance_williams(self, r_us, r_vs, r_uv, u_len, v_len, s_len, method='min'):
#         """Apply the Lance-Williams formula to compute the new distance between merged clusters."""
#         au = av = b = g = 0

#         if method == 'min':  # Single linkage (minimum distance)
#             au = 0.5
#             av = 0.5
#             b = 0
#             g = -0.5
#         elif method == 'max':  # Complete linkage (maximum distance)
#             au = 0.5
#             av = 0.5
#             b = 0
#             g = 0.5
#         elif method == 'mean':  # Average linkage (mean distance)
#             au = u_len / (u_len + v_len)
#             av = v_len / (u_len + v_len)
#             b = g = 0
#         elif method == 'center':  # Centroid linkage
#             au = u_len / (u_len + v_len)
#             av = v_len / (u_len + v_len)
#             b = -au * av
#             g = 0
#         elif method == 'ward':  # Ward's method
#             au = (s_len + u_len) / (s_len + u_len + v_len)
#             av = (s_len + v_len) / (s_len + u_len + v_len)
#             b = -s_len / (s_len + u_len + v_len)
#             g = 0
#         else:
#             raise Exception("Invalid method. Choose one of: 'min', 'max', 'mean', 'center', 'ward'.")

#         return au * r_us + av * r_vs + b * r_uv + g * np.abs(r_us - r_vs)

#     def fit(self, X):
#         # Starting with each point as its own cluster
#         n_samples = X.shape[0]
#         clusters = {i: [i] for i in range(n_samples)}  # Track clusters
#         distances = squareform(pdist(X))  # Pairwise distance calculation
#         np.fill_diagonal(distances, np.inf)  # Set diagonal to infinity (no self-merge)
#         while len(clusters) > self.n_clusters:
#             # Find the closest pair of clusters
#             min_dist_idx = np.unravel_index(np.argmin(distances), distances.shape)
#             c1, c2 = min_dist_idx
#             dist = distances[c1, c2]

#             # Merge clusters c1 and c2 BEFORE appending to linkage matrix
#             clusters[c1].extend(clusters[c2])

#             # Append to linkage matrix (after merging, with correct size)
#             merged_cluster_size = len(clusters[c1])  # Now the size is correct
#             self.linkage_matrix.append([c1, c2, dist, merged_cluster_size])

#             # Now delete the second cluster
#             del clusters[c2]

#             # Update distances using Lance-Williams formula
#             for i in range(n_samples):
#                 if i != c1 and i in clusters:
#                     # Current distances between cluster i and merged cluster c1/c2
#                     r_us = distances[c1, i]
#                     r_vs = distances[c2, i]
#                     r_uv = distances[c1, c2]
#                     u_len = len(clusters[c1])
#                     v_len = len(clusters[i])  # Use len(clusters[i]), don't reference c2
#                     s_len = u_len + v_len

#                     # Update distance using the Lance-Williams formula
#                     distances[c1, i] = distances[i, c1] = self.lance_williams(
#                         r_us, r_vs, r_uv, u_len, v_len, s_len, method=self.linkage
#                     )

#             # Set distances of the removed cluster to infinity
#             distances[:, c2] = distances[c2, :] = np.inf


#         # Assign labels to points
#         self.labels_ = np.full(n_samples, -1)
#         for cluster_id, cluster_points in clusters.items():
#             for point in cluster_points:
#                 self.labels_[point] = cluster_id
#         return clusters

#     def fit_predict(self, X):
#         self.fit(X)
#         return self.labels_

#     def plot_clusters(self, X):
#         """Plot the clusters."""
#         if self.labels_ is None:
#             raise ValueError("You must fit the model before plotting.")
        
#         plt.figure(figsize=(8, 6))
#         distinct_colors = ['red', 'blue', 'green', 'purple', 'orange', 'brown', 'pink', 'gray', 'cyan', 'yellow']
#         color_count = len(distinct_colors)
#         unique_labels = np.unique(self.labels_)

#         label_to_color = {label: distinct_colors[idx % color_count] for idx, label in enumerate(unique_labels)}
        
#         for cluster_id in unique_labels:
#             cluster_points = X[self.labels_ == cluster_id]
#             color = label_to_color[cluster_id]
#             plt.scatter(cluster_points[:, 0], cluster_points[:, 1], 
#                         color=color, label=f'Cluster {cluster_id}')
        
#         plt.title(f'Hierarchical Clustering (n_clusters={self.n_clusters}, linkage={self.linkage})')
#         plt.xlabel('Feature 1')
#         plt.ylabel('Feature 2')
#         plt.legend()
#         plt.grid(True)
#         plt.show()

    
# def plot_all_linkage_methods(X, n_clusters=2):
#     """Plot hierarchical clustering with different linkage methods using unique colors for each cluster."""
#     linkage_methods = ['min', 'max', 'mean', 'center', 'ward']
#     fig, axs = plt.subplots(1, len(linkage_methods), figsize=(20, 5))

#     # Define a large set of distinct colors for clusters
#     distinct_colors = ['red', 'blue', 'green', 'purple', 'orange', 'brown', 
#                        'pink', 'gray', 'cyan', 'yellow', 'magenta', 'black', 
#                        'lime', 'teal', 'olive', 'navy', 'maroon', 'gold']

#     # Extend colors if needed by repeating the list (you can add more unique colors if needed)
#     color_count = len(distinct_colors)
    
#     for i, method in enumerate(linkage_methods):
#         model = my_HierarchicalClustering(n_clusters=n_clusters, linkage=method)
#         labels = model.fit_predict(X)
        
#         # Get the unique cluster labels and map each label to a unique color
#         unique_labels = np.unique(labels)
#         label_to_color = {label: distinct_colors[idx % color_count] for idx, label in enumerate(unique_labels)}
        
#         for cluster_id in unique_labels:
#             cluster_points = X[labels == cluster_id]
#             color = label_to_color[cluster_id]  # Use the consistent color for each cluster ID
#             axs[i].scatter(cluster_points[:, 0], cluster_points[:, 1], 
#                            color=color, label=f'Cluster {cluster_id}')

#         axs[i].set_title(f'Linkage: {method}')
#         axs[i].grid(True)

#     plt.tight_layout()
#     plt.show()
#     return labels

# df = pd.read_csv('data/Mall_Customers.csv')[['Annual Income (k$)', 'Spending Score (1-100)']]
# h = my_HierarchicalClustering(n_clusters = 5, linkage='ward')

# labels = h.fit_predict(df.values)

# plot_all_linkage_methods(df.values,5)
