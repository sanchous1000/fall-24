import numpy as np
from scipy.stats import multivariate_normal


class MyEM:
    def __init__(self, n_clusters: int = 2, max_iter: int = 100, weight_delta_threshold: float = 1e-3):
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.weight_threshold = weight_delta_threshold
        self.means = None
        self.si = None
        self.weights = None

    def fit_predict(self, X):
        X = X.values
        n_samples, n_features = X.shape

        self.means = X[np.random.choice(n_samples, self.n_clusters, replace=False)]
        self.si = [np.cov(X.T) for _ in range(self.n_clusters)]
        self.weights = np.ones(self.n_clusters) / self.n_clusters
        probabilities = np.zeros((n_samples, self.n_clusters))

        for iteration in range(self.max_iter):
            for k in range(self.n_clusters):
                probabilities[:, k] = self.weights[k] * multivariate_normal.pdf(
                    X,
                    mean=self.means[k],
                    cov=self.si[k],
                    allow_singular=True,
                )

            probabilities /= probabilities.sum(axis=1, keepdims=True)

            N_k = probabilities.sum(axis=0)

            new_means = np.dot(probabilities.T, X) / N_k[:, np.newaxis]
            new_covariances = []

            for k in range(self.n_clusters):
                if N_k[k] == 0:
                    new_covariances.append(np.eye(n_features))
                else:
                    diff = X - new_means[k]
                    new_covariances.append(np.dot(probabilities[:, k] * diff.T, diff) / N_k[k])

            new_weights = N_k / n_samples

            if (np.abs(new_weights - self.weights) < self.weight_threshold).all():
                break

            self.means = new_means
            self.si = new_covariances
            self.weights = new_weights

        return np.argmax(probabilities, axis=1)
