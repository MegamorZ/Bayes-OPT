import numpy as np
import matplotlib.pyplot as plt
import itertools
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


def bounded_min(Func, bounds, method="L-BFGS-B", n_restarts=10, *args):
    Fmin = None
    Xmin = None
    ndim = bounds.shape[1]

    for _ in range(n_restarts):
        # Creo punto al azar dentro de los limites
        X0 = np.array([np.random.uniform(low=b[0], high=b[1]) for b in bounds])
        # Busco minimo
        res = minimize(Func, x0=X0, args=args, method=method, bounds=bounds)

        if Fmin is None or res.fun < Fmin:
            Xmin = res.x
            Fmin = res.fun
    return Xmin, Fmin


class BayOptRBF:
    def __init__(self, X_train, y_train, alpha=0.1):
        # Guardo los datos de entrenamiento/ parametros
        self.X_train = X_train
        self.n_dim = self.X_train.shape[1]
        self.y_train = y_train
        self.alpha = alpha
        self.y_train_max = max(self.y_train)

        # Limites para el hyperparametro lengthscale
        _dist = pdist(self.X_train)
        _dist_max = np.max(_dist)
        _dist_min = np.min(_dist) / 2 if np.min(_dist) > 0.5 else 0.5

        # Inicializo el kernel
        self.kernel = RBF(
            length_scale_bounds=(_dist_min, _dist_max)
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
                acq_fun_max = -res.fun
                if acq_fun_max > current_acq_fun_max:
                    current_acq_fun_max = acq_fun_max
                    optimal_X = res.x

        return optimal_X, current_acq_fun_max[0]

    def fit(self, X_train, y_train):
        self.regressor.fit(X_train, y_train)


class RandomQuadratic:
    # uso definicion matricial https://en.wikipedia.org/wiki/Quadratic_form
    # definidas/semidefinidas negativas para encontrar maximos.

    def __init__(self, n_dim, bounds=None, offset=True, noise=False):
        self.n_dim = n_dim
        self.offset = offset
        self.noise = noise
        self.bounds = bounds

        # genero matriz asociada al polinomio al azar

        # Semidefinidas negativas
        # random_matrix = np.random.uniform(low=-1, high=0,size=(n_dim,n_dim))
        # self.associated_matrix = (random_matrix+random_matrix.T)/2

        # definidas negativas
        random_vec = np.random.uniform(low=-1, high=-0.1, size=self.n_dim)
        self.associated_matrix = np.diag(random_vec)

        # si offset=True -> cambio la pos del centro
        # si offset=False -> maximo en zero
        if offset:
            self.x0 = np.random.uniform(low=-1, high=1, size=(1, self.n_dim))

        if self.bounds is not None:
            bound_points = np.array(
                [pairs for pairs in itertools.product(*self.bounds)]
            )
            self.bounded_min = np.min(
                [self.__call__(p, normalize=False) for p in bound_points]
            )
            self.bounded_max = (
                self.__call__(self.x0, normalize=False)
                if offset
                else self.__call__(np.zeros(shape=self.n_dim), normalize=False)
            )

    def __call__(self, x, normalize=True):
        # X un punto representado por un matriz (1xN)
        if self.offset:
            x = x - self.x0
        # valor del polinomio en el punto x
        y = float(np.dot(x, np.dot(self.associated_matrix, x.T)))
        # Escalo valores entre 0 y 1 dentro de los limites
        if normalize:
            y = (y - self.bounded_min) / (self.bounded_max - self.bounded_min)

        # Agregar ruido gaussiano
        if self.noise:
            return y + np.random.normal(0, self.noise)
        else:
            return y


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

    def __call__(self, x):
        # X un punto representado por un matriz (1xN) //[[x1,xi,xn]]
        output = np.array([])
        for xs, x0, c in zip(x, self.x0, self.c):
            components = np.append(output, np.exp(-((xs - x0) ** 2) / (2 * c**2)))
        return np.prod(components) * self.scale_factor


def plot_3d(RS, regressor):
    # figura
    fig, ax = plt.subplots(1, 2, figsize=(10, 10), subplot_kw=dict(projection="3d"))

    # mesh
    x = y = np.linspace(-1, 1, 20)
    X, Y = np.meshgrid(x, y)
    # convierto a coordenadas y calculo el valor en cada punto
    z = np.array([])
    z_aprox = np.array([])
    for x0, y0 in zip(np.ravel(X), np.ravel(Y)):
        z = np.append(z, RS(np.array([[x0, y0]])))
        z_aprox = np.append(z_aprox, regressor.predict(np.array([[x0, y0]])))

    # convierto los valores a forma de mesh y ploteo
    Z = z.reshape(X.shape)
    Z_aprox = z_aprox.reshape(X.shape)

    ax[0].plot_surface(X, Y, Z)

    ax[1].plot_surface(X, Y, Z_aprox)

    ax[0].set_xlabel("X Label")
    ax[0].set_ylabel("Y Label")
    ax[0].set_zlabel("Z Label")

    ax[1].set_xlabel("X Label")
    ax[1].set_ylabel("Y Label")
    ax[1].set_zlabel("Z Label")

    ax[0].set_title("Ground Truth")
    ax[1].set_title("Aproximation")
    plt.show()
