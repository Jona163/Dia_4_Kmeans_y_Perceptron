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
        
    # Método para entrenar el perceptrón
    def fit(self, X, y):
        n_samples, n_features = X.shape  # Obtener el número de muestras y características

        # Inicializar los pesos y el sesgo a 0
        self.weights = np.zeros(n_features)
        self.bias = 0

        # Asegurar que las etiquetas sean 0 o 1
        y_ = np.where(y > 0 , 1, 0)

        # Aprendizaje de los pesos
        for _ in range(self.n_iters):
            for idx, x_i in enumerate(X):
                # Cálculo de la salida lineal
                linear_output = np.dot(x_i, self.weights) + self.bias
                y_predicted = self.activation_func(linear_output)

                # Regla de actualización del perceptrón
                update = self.lr * (y_[idx] - y_predicted)
                self.weights += update * x_i  # Actualización de los pesos
                self.bias += update           # Actualización del sesgo
    
    # Método para predecir
    def predict(self, X):
        # Cálculo de la salida lineal y predicción
        linear_output = np.dot(X, self.weights) + self.bias
        y_predicted = self.activation_func(linear_output)
        return y_predicted

# Pruebas del perceptrón
if __name__ == "__main__":
    # Imports
    import matplotlib.pyplot as plt
    from sklearn.model_selection import train_test_split
    from sklearn import datasets

    # Función para calcular la precisión de las predicciones
    def accuracy(y_true, y_pred):
        accuracy = np.sum(y_true == y_pred) / len(y_true)
        return accuracy

    # Generar un conjunto de datos de prueba
    X, y = datasets.make_blobs(
        n_samples=150, n_features=2, centers=2, cluster_std=1.05, random_state=2
    )
    
    # Dividir los datos en entrenamiento y prueba
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=123
    )

    # Inicializar y entrenar el perceptrón
    p = Perceptron(learning_rate=0.01, n_iters=1000)
    p.fit(X_train, y_train)

    # Predecir sobre los datos de prueba
    predictions = p.predict(X_test)

    # Imprimir la precisión del perceptrón
    print("Precisión de la clasificación del Perceptrón:", accuracy(y_test, predictions))

    # Visualización de los resultados
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    
    # Graficar las muestras de entrenamiento
    plt.scatter(X_train[:, 0], X_train[:, 1], marker="o", c=y_train)

    # Calcular la línea de decisión del perceptrón
    x0_1 = np.amin(X_train[:, 0])
    x0_2 = np.amax(X_train[:, 0])

    x1_1 = (-p.weights[0] * x0_1 - p.bias) / p.weights[1]
    x1_2 = (-p.weights[0] * x0_2 - p.bias) / p.weights[1]

    # Graficar la línea de decisión
    ax.plot([x0_1, x0_2], [x1_1, x1_2], "k")

    # Ajustar los límites del eje y para mejor visualización
    ymin = np.amin(X_train[:, 1])
    ymax = np.amax(X_train[:, 1])
    ax.set_ylim([ymin - 3, ymax + 3])

    # Mostrar la gráfica
    plt.show()

