import random
import numpy as np

random.seed(52)


def expectation_maximization(data, num_features, num_clusters, max_iterations=30):
    num_samples = len(data)

    cluster_centers = np.array([random.choice(data) for _ in range(num_clusters)])

    variances = np.empty((num_clusters, num_features))
    for cluster_idx in range(num_clusters):
        differences = data - cluster_centers[cluster_idx]
        variances[cluster_idx] = np.diag(np.dot(differences.T, differences)) / num_samples

    cluster_weights = np.full((num_clusters, 1), 1 / num_clusters)

    previous_labels = None
    for iteration in range(max_iterations):
        probabilities = []
        for cluster_idx in range(num_clusters):
            dist_squared = np.sum((data - cluster_centers[cluster_idx]) ** 2 / variances[cluster_idx], axis=1)
            norm_factor = (2 * np.pi) ** (-num_features / 2) / np.prod(np.sqrt(variances[cluster_idx]))
            probabilities.append(norm_factor * np.exp(-0.5 * dist_squared))
        probabilities = np.array(probabilities).T

        responsibilities = cluster_weights.T * probabilities
        responsibilities /= responsibilities.sum(axis=1, keepdims=True)

        cluster_weights = responsibilities.sum(axis=0) / num_samples
        cluster_centers = (responsibilities.T @ data) / cluster_weights[:, np.newaxis] / num_samples
        variances = np.empty((num_clusters, num_features))
        for cluster_idx in range(num_clusters):
            weighted_differences = responsibilities[:, cluster_idx] @ (data - cluster_centers[cluster_idx]) ** 2
            variances[cluster_idx] = weighted_differences / cluster_weights[cluster_idx] / num_samples

        current_labels = np.argmax(responsibilities, axis=1)
        if previous_labels is not None and np.array_equal(current_labels, previous_labels):
            return current_labels
        previous_labels = current_labels

    return current_labels
