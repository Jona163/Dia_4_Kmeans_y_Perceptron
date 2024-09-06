# Autor: Jonathan Hernández
# Fecha: 05 Septiembre 2024
# Descripción: Código para una simulación de los Kmeans
# GitHub: https://github.com/Jona163

#Importacion de las librerias 
import numpy as np
import matplotlib.pyplot as plt

def euclidean_distance(x1, x2):
    return np.sqrt(np.sum((x1-x2)**2))

class KMeans:

    def __init__(self, K=5, max_iters=100, plot_steps=False):
        self.K = K
        self.max_iters = max_iters
        self.plot_steps = plot_steps

        # lista de sample de indices para cada cluster
        self.clusters = [[] for _ in range(self.K)]

        #  El centro de los (mean vector) para cada cluster
        self.centroids = []

     def predict(self, X):
        self.X = X
        self.n_samples, self.n_features = X.shape

        # inicializaciòn
        random_sample_idxs = np.random.choice(self.n_samples, self.K, replace=False)
        self.centroids = [self.X[idx] for idx in random_sample_idxs]

        # optimizacion de clusters
        for _ in range(self.max_iters):
            # assign samples to closest centroids (create clusters)
            self.clusters = self._create_clusters(self.centroids)

            if self.plot_steps:
                self.plot()

            # calcular nuevos centroides para el clusters
            centroids_old = self.centroids
            self.centroids = self._get_centroids(self.clusters)

            if self._is_converged(centroids_old, self.centroids):
                break

            if self.plot_steps:
                self.plot()

        # clasificar los samples de el indice para los clusters
        return self._get_cluster_labels(self.clusters)
