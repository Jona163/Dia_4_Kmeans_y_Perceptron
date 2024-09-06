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
