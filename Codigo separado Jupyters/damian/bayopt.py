import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF
from scipy.optimize import minimize
from scipy.spatial.distance import pdist


class BayOptRBF:
    def __init__(self, X_train, y_train, alpha=0.1):
        # Guardo los datos de entrenamiento/ parametros
        self.X_train = X_train
        self.y_train = y_train
        self.alpha = alpha  # varianza de ruido gaussiano adicional a las mediciones

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

