import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF
from scipy.optimize import minimize
from scipy.spatial.distance import pdist
from scipy.stats import norm

# Funciones de adquisicion.
# Fuente: https://ekamperi.github.io/machine%20learning/2021/06/11/acquisition-functions.html


def EI(x, gp, y_max, epsilon=0):
    """funcion de adquisicion - expected improvement

    Args:
        x (np.array): valor de prueba en forma de vector.
        gp (GaussianProcessRegressor): Modelo
        y_max (float): valor maximo del set de entrenamiento.
        epsilon (int, optional): valores positivos favorecen exploracion vs explotacion, default=0.

    Returns:
        float: expecativa de mejora
    """

    # Calcular media y desvio para el valor de prueba "x"
    mean, std = gp.predict(x, return_std=True)

    # calcular y devolver la expectativa de mejora
    a = mean - y_max - epsilon
    z = a / std

    return a * norm.cdf(z) + std * norm.pdf(z)


def PI(x, gp, y_max, epsilon=0):
    """funcion de adquisicion - probability of improvement

    Args:
        x (np.array): valor de prueba en forma de vector.
        gp (GaussianProcessRegressor): Modelo
        y_max (float): valor maximo del set de entrenamiento.
        epsilon (int, optional): valores positivos favorecen exploracion vs explotacion, default=0.

    Returns:
        float: probabilidad de mejora
    """

    # Calcular media y desvio para el valor de prueba "x"
    mean, std = gp.predict(x, return_std=True)

    # calcular y devolver la probabilidad de mejora

    z = mean - y_max - epsilon / std

    return norm.cdf(z)


def UCB(x, gp, kappa=1.96):
    """funcion de adquisicion - upper confidence bound

    Args:
        x (np.array): valor de prueba en forma de vector.
        gp (GaussianProcessRegressor): Modelo
        kappa (int, optional): mayores valores favorecen exploracion vs explotacion, default=1.

    Returns:
        float: mean + kappa * std
    """

    # Calcular media y desvio para el valor de prueba "x"
    mean, std = gp.predict(x, return_std=True)

    # calcular y devolver la probabilidad de mejora

    return mean + kappa * std


class BayOptRBF:
    def __init__(self, X_train, y_train, alpha=0.1):
        # Guardo los datos de entrenamiento/ parametros
        self.X_train = X_train
        self.y_train = y_train
        self.alpha = alpha  # varianza del ruido gaussiano adicionado a los datos

        # Inicializo el kernel
        self.kernel = (
            RBF()
        )  # ver necesidad de setear limites al hiper parametro lengthscale

        # Inicializo el regresor y ajusto a los datos.
        self.regressor = GaussianProcessRegressor(
            kernel=self.kernel,
            n_restarts_optimizer=9,
            alpha=self.alpha,
            normalize_y=True,
        )
        self.regressor.fit(self.X_train, self.y_train)

    def global_max(self, acq_fun="EI", n_restarts=20):
        pass
