# Autor: Jonathan Hernández
# Fecha: 05 Septiembre 2024
# Descripción: Código para una simulación de Perceptron.
# GitHub: https://github.com/Jona163

#Importancion de liberias 
import numpy as np

# Función escalón unitario (unit step function)
# Retorna 1 si x > 0, de lo contrario retorna 0.
def unit_step_func(x):
    return np.where(x > 0 , 1, 0)

# Clase Perceptron
class Perceptron:

    # Inicialización del perceptrón con tasa de aprendizaje y número de iteraciones
    def __init__(self, learning_rate=0.01, n_iters=1000):
        self.lr = learning_rate  # Tasa de aprendizaje
        self.n_iters = n_iters   # Número de iteraciones
        self.activation_func = unit_step_func  # Función de activación (escalón unitario)
        self.weights = None      # Pesos (inicialmente indefinidos)
        self.bias = None         # Sesgo (inicialmente indefinido)
