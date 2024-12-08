import numpy as np
from scipy.spatial.distance import cdist
import random


def dbscan(data, eps, min_points):
    dist_matrix = cdist(data, data)
    np.fill_diagonal(dist_matrix, np.inf)

    num_points = len(data)
    cluster_labels = np.full(num_points, np.nan)
    point_roles = np.full(num_points, np.nan)
    cluster_id = 0
    iteration_count = 0

    point_indices = np.arange(num_points, dtype=np.uint32)

    while np.isnan(point_roles).any():
        unvisited = np.where(np.isnan(cluster_labels))[0]
        seed_point = random.choice(unvisited)

        neighbors = point_indices[dist_matrix[seed_point] < eps]
        if len(neighbors) < min_points:
            point_roles[seed_point] = -1
            candidates = [seed_point]
        else:
            point_roles[seed_point] = 1
            cluster_labels[seed_point] = cluster_id
            candidates = np.append(neighbors, seed_point)

            while np.isnan(cluster_labels[candidates]).any():
                pending_points = candidates[np.isnan(cluster_labels[candidates])]
                current_point = random.choice(pending_points)

                current_neighbors = point_indices[dist_matrix[current_point] < eps]
                if len(current_neighbors) >= min_points:
                    point_roles[current_point] = 1
                    candidates = np.union1d(candidates, current_neighbors)
                else:
                    point_roles[current_point] = 0
                cluster_labels[current_point] = cluster_id

            cluster_id += 1

        dist_matrix[candidates] = np.inf
        iteration_count += 1
        if iteration_count >= 500:
            break

    cluster_labels[np.isnan(cluster_labels)] = -1
    return point_roles, cluster_labels
