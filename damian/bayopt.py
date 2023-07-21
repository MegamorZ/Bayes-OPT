from typing import Any
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF
from scipy.optimize import minimize
from scipy.spatial.distance import pdist
from scipy.stats import norm

# Funciones de adquisicion.
# Fuente: https://ekamperi.github.io/machine%20learning/2021/06/11/acquisition-functions.html
# OBS> regresan el valor cambiado de signo para utilizar scipy.optimize.minize para buscar maximos.


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
    mean, std = gp.predict([x], return_std=True)

    # calcular y devolver la expectativa de mejora
    a = mean - y_max - epsilon
    z = a / std

    result = a * norm.cdf(z) + std * norm.pdf(z)
    return -result


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
    mean, std = gp.predict([x], return_std=True)

    # calcular y devolver la probabilidad de mejora

    z = mean - y_max - epsilon / std

    result = norm.cdf(z)
    return -result


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
    mean, std = gp.predict([x], return_std=True)

    # calcular y devolver la probabilidad de mejora

    result = mean + kappa * std
    return -result


class BayOptRBF:
    def __init__(self, X_train, y_train, alpha=0.01):
        # Guardo los datos de entrenamiento/ parametros
        self.X_train = X_train
        self.n_dim = self.X_train.shape[1]
        self.y_train = y_train
        self.alpha = alpha  # varianza del ruido gaussiano adicionado a los datos
        self.y_train_max = max(self.y_train)

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

    def global_max(self, bounds, acq_fun="EI", n_restarts=20):
        ## bounds [(min,max), ...] min y max por cada variable

        # seleccion de funcion de adquisicion y parametros para pasar al optimizador
        if acq_fun == "EI":
            fun = EI
            params = (self.regressor, self.y_train_max)
        if acq_fun == "PI":
            fun = PI
            params = (self.regressor, self.y_train_max)
        if acq_fun == "UCB":
            fun = UCB
            params = self.regressor

        # busco un maximo de la funcion de adquisicion iterativamente para
        # distintos puntos iniciales elegidos al azar.
        current_acq_fun_max = 0
        optimal_X = None

        for _ in range(n_restarts):
            # elijo puntos al azar dentro de los limites
            x0 = np.array([np.random.uniform(min, max) for min, max in bounds])

            # Busco maximos con minimize (acq_fun cambiadas de signo)
            res = minimize(fun, x0, args=params, method="L-BFGS-B", bounds=bounds)

            if res.success:  # Si converge el optimizador
                acq_fun_max = -res.fun[0]
                if acq_fun_max > current_acq_fun_max:
                    current_acq_fun_max = acq_fun_max
                    optimal_X = res.x

        return optimal_X, current_acq_fun_max

class RandomQuadratic:
    #uso definicion matricial https://en.wikipedia.org/wiki/Quadratic_form
    #definidas/semidefinidas negativas para encontrar maximos. 

    def __init__(self, n_dim, scale_factor=1, offset=True):
        self.n_dim = n_dim
        self.scale_factor = scale_factor
        self.offset = offset


        #genero matriz asociada al polinomio al azar 
        random_matrix = np.random.uniform(low=-1, high=0,size=(n_dim,n_dim))
        self.associated_matrix = (random_matrix+random_matrix.T)/2
        
        #si offset=True -> cambio la pos del maximo
        #si offset=False -> maximo en zero
        if offset:
            self.x0 = np.random.uniform(low=-1, high=1, size=(1, n_dim))

    def __call__(self,x):
        #X un punto representado por un matriz (1xN)
        if self.offset:
            x = x-self.x0
        return float(np.dot(x, np.dot(self.associated_matrix, x.T)))

class RandomGaussian:
    def __init__(self, n_dim, scale_factor=1.0, offset=True):
        self.n_dim = n_dim
        self.scale_factor = scale_factor
        self.offset = offset
        # G~EXP(-(x-x0)**2/2*c**2)
        if offset:
            self.x0 = np.random.uniform(low=-1, high=1, size=(n_dim))
        else:
            self.x0 = np.zeros(shape=(n_dim))

        self.c = np.random.uniform(low=0.2, high=1, size=(n_dim))    

    def __call__(self,x):
        #X un punto representado por un matriz (1xN) //[[x1,xi,xn]]
        output = np.array([])
        for xs, x0, c in zip(x,self.x0, self.c):
            components = np.append(output, np.exp(-((xs-x0)**2)/(2*c**2)))
        return np.prod(components)*self.scale_factor

