import numpy as np
from scipy.spatial.distance import cdist


def inner_cluster_distance(data, labels):
    distances = []
    unique_labels = np.unique(labels)

    for label_a in unique_labels:
        for label_b in unique_labels:
            if label_b > label_a:
                cluster_a = data[labels == label_a]
                cluster_b = data[labels == label_b]
                distance = (
                        np.sum(cdist(cluster_a, cluster_b)) *
                        len(cluster_a) * len(cluster_b) /
                        (len(cluster_a) + len(cluster_b))
                )
                distances.append(distance)

    return np.mean(distances)


def outer_cluster_distance(data, labels):
    distances = []
    unique_labels = np.unique(labels)

    for label in unique_labels:
        cluster_points = data[labels == label]
        cluster_center = np.mean(cluster_points, axis=0).reshape(1, -1)
        intra_distance = np.sum(cdist(cluster_points, cluster_center))
        distances.append(intra_distance)

    return np.mean(distances)
