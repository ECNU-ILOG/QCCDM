import numpy as np


def mean_avg_distance(X):
    n = X.shape[0]
    dist = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            dist[i, j] = np.linalg.norm(X[i] - X[j], 2)
            dist[j, i] = dist[i, j]
    return np.mean(dist)
