from collections import deque

import numpy as np


class MyDBSCAN:
    def __init__(self, eps=3, min_samples=3, metric='euclidean'):
        self.eps = eps
        self.min_samples = min_samples
        self.metric = metric

    def __repr__(self) -> str:
        return f'MyDBSCAN class: eps={self.eps}, min_samples={self.min_samples}'

    def fit_predict(self, X):
        X = np.array(X)
        observations, feature_count = X.shape
        roots = {}
        border = set()
        outliers = set()

        core_to_cluster = {}

        neighbours = self._get_neighbours(X)
        for i in range(observations):
            if i in outliers or i in border:
                continue

            if len(neighbours[i]) < self.min_samples:
                outliers.add(i)
                continue

            if i not in core_to_cluster:
                core_to_cluster[i] = i

            cluster = core_to_cluster[i]

            if cluster not in roots:
                roots[cluster] = set()

            traversal_list = deque(neighbours[i])
            seen = set()
            while traversal_list:
                neighbour = traversal_list.popleft()
                if neighbour in seen:
                    continue
                seen.add(neighbour)
                if len(neighbours[neighbour]) < self.min_samples:
                    border.add(neighbour)
                else:
                    core_to_cluster[neighbour] = cluster
                    traversal_list.extend(neighbours[neighbour])
                roots[cluster].add(neighbour)


        res = [0] * observations

        # outliers go to cluster -1
        for ol in outliers:
            res[ol] = -1

        for cluster_root, points in roots.items():
            for point in points:
                res[point] = cluster_root

        return res

    def _get_neighbours(self, X):
        neighbours = [set() for _ in range(X.shape[0])]
        for i in range(X.shape[0]):
            for j in range(X.shape[0]):
                if i == j:
                    continue

                dist = self._get_dist(X[i], X[j])
                if dist < self.eps:
                    neighbours[i].add(j)

        return neighbours

    def _get_dist(self, x, y):
        if self.metric == 'euclidean':
            return np.linalg.norm(x - y)
        if self.metric == 'chebyshev':
            return np.max(np.abs(x - y))
        if self.metric == 'manhattan':
            return np.sum(np.abs(x - y))
        if self.metric == 'cosine':
            return 1 - np.sum(np.multiply(x, y)) / (np.linalg.norm(x) * np.linalg.norm(y))