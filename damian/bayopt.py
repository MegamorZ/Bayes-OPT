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
    def __init__(self, n_dim, offset=True, noise=False):
        self.n_dim = n_dim
        self.offset = offset
        self.noise = noise
        # G~EXP(-(x-x0)**2/2*c**2)
        if offset:
            self.x0 = np.random.uniform(low=-0.5, high=0.5, size=(n_dim))
        else:
            self.x0 = np.zeros(shape=(n_dim))

        self.c = np.random.uniform(low=0.2, high=1, size=(n_dim))

    def __call__(self, x):
        # X un punto representado por un matriz (1xN) //[[x1,xi,xn]]
        output = np.array([])
        for xs, x0, c in zip(x, self.x0, self.c):
            components = np.append(output, np.exp(-((xs - x0) ** 2) / (2 * c**2)))
        y = np.prod(components) + 1

        # Agregar ruido gaussiano
        if self.noise:
            return y + np.random.normal(0, self.noise)
        else:
            return y


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


class SimplexOpt:
    """Metodo simplex explicado en:
    M.A.Bezerra etal. / MicrochemicalJournal124 (2016) 45"""

    def __init__(self, bounds, step=1):
        self.bounds = bounds
        self.step = step  # step
        self.x0 = np.array(
            [np.random.uniform(low=bound[0], high=bound[1]) for bound in self.bounds]
        )  # starting point
        self.n_dim = self.bounds.shape[0]  # dimension, debe ser menor a 10
        self.SIV = np.array(
            [
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0.5, 0.87, 0, 0, 0, 0, 0, 0, 0, 0],
                [0.5, 0.29, 0.82, 0, 0, 0, 0, 0, 0, 0],
                [0.5, 0.29, 0.2, 0.79, 0, 0, 0, 0, 0, 0],
                [0.5, 0.29, 0.2, 0.16, 0.78, 0, 0, 0, 0, 0],
                [0.5, 0.29, 0.2, 0.16, 0.13, 0.76, 0, 0, 0, 0],
                [0.5, 0.29, 0.2, 0.16, 0.13, 0.11, 0.76, 0, 0, 0],
                [0.5, 0.29, 0.2, 0.16, 0.13, 0.11, 0.094, 0.75, 0, 0],
                [0.5, 0.29, 0.2, 0.16, 0.13, 0.11, 0.094, 0.083, 0.75, 0],
            ]
        )
        self.current_vertex = self.initial_vertex()

    def initial_vertex(self):
        # Vertex inicial
        x_origin = np.array(
            [
                self.x0,
            ]
            * ((self.n_dim) + 1)
        )
        vertex = x_origin + self.SIV[0 : (self.n_dim) + 1, 0 : self.n_dim] * self.step
        return vertex

    def simplex(self, x_vertex_data, y_vertex_data):
        # devuelve el proximo punto a muestrear
        # actualiza self.current_vertex con el nuevo vertex

        index_of_min = np.argmin(y_vertex_data)

        # W es el peor valor del vertex, M es el punto central del vertex excluido W.
        W = x_vertex_data[index_of_min]
        menos_W = np.delete(x_vertex_data, index_of_min, axis=0)
        M = menos_W.mean(axis=0)
        R = 2 * M - W

        self.current_vertex = np.concatenate((menos_W, [R]))
        return R


class SimplexModOpt:
    """Metodo simplex modificado explicado en:
    M.A.Bezerra etal. / MicrochemicalJournal124 (2016) 45"""

    def __init__(self, bounds, step=1):
        self.bounds = bounds
        self.step = step  # step
        self.x0 = np.array(
            [np.random.uniform(low=bound[0], high=bound[1]) for bound in self.bounds]
        )  # starting point
        self.n_dim = self.bounds.shape[0]  # dimension, debe ser menor a 10
        self.SIV = np.array(
            [
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0.5, 0.87, 0, 0, 0, 0, 0, 0, 0, 0],
                [0.5, 0.29, 0.82, 0, 0, 0, 0, 0, 0, 0],
                [0.5, 0.29, 0.2, 0.79, 0, 0, 0, 0, 0, 0],
                [0.5, 0.29, 0.2, 0.16, 0.78, 0, 0, 0, 0, 0],
                [0.5, 0.29, 0.2, 0.16, 0.13, 0.76, 0, 0, 0, 0],
                [0.5, 0.29, 0.2, 0.16, 0.13, 0.11, 0.76, 0, 0, 0],
                [0.5, 0.29, 0.2, 0.16, 0.13, 0.11, 0.094, 0.75, 0, 0],
                [0.5, 0.29, 0.2, 0.16, 0.13, 0.11, 0.094, 0.083, 0.75, 0],
            ]
        )
        self.current_vertex = self.initial_vertex()

        # Estado para modificar contraccion/expansion del vertex
        self.last_vertex = None
        self.last_response = None
        self.last_point_added = None
        self.last_point_added_type = None

    def initial_vertex(self):
        # Vertex inicial
        x_origin = np.array(
            [
                self.x0,
            ]
            * ((self.n_dim) + 1)
        )
        vertex = x_origin + self.SIV[0 : (self.n_dim) + 1, 0 : self.n_dim] * self.step

        return vertex

    def simplex(self, x_vertex_data, y_vertex_data):
        # devuelve el proximo punto a muestrear
        # actualiza self.current_vertex con el nuevo vertex

        index_of_min = np.argmin(y_vertex_data)

        # W es el peor valor del vertex, B es el mejor valor del vertex, M es el punto central del vertex excluido W.
        W = x_vertex_data[index_of_min]

        menos_W = np.delete(x_vertex_data, index_of_min, axis=0)
        M = menos_W.mean(axis=0)
        # R refleccion, E refleccion expandida, CR refleccion contraida, CW cambio de direccion.
        R = 2 * M - W

        # en la primera iteracion devuelvo R
        if not self.last_vertex:
            # guardo el vertex inicial
            self.last_vertex = self.x_vertex_data
            self.last_response = self.y_vertex_data
            self.last_point_added = R
            self.last_point_added_type = "R"

            self.current_vertex = np.concatenate((menos_W, [R]))
            return R

        # En la segunda iteracion, modifico el output segun la respuesta

        # Si el ultimo punto agregado fue de tipo "R"
        if self.last_point_added_type == "R":
            # Caso 1: Señal en R fue mejor que señal en B del vertex anterior -> expansion
            B_ = self.last_response.max()
            R_ = self.y_vertex_data[
                np.where(self.x_vertex == self.last_point_added)[0][0]
            ]

            if R_ > B_:
                self.last_vertex = self.x_vertex_data
                self.last_response = self.y_vertex_data
                E = 3 * M - 2 * W
                self.last_point_added = E
                self.last_point_added_type = "E"

                self.current_vertex = np.concatenate((menos_W, [E]))
                return E
            # Caso 2,   N < R < B --> mantengo  BNR simplex
            N_ = self.last_response[np.argsort(self.last_response)[-2]]

            if N_ < R_ < B_:
                self.last_vertex = self.x_vertex_data
                self.last_response = self.y_vertex_data
                self.last_point_added = R
                self.last_point_added_type = "R"

                self.current_vertex = np.concatenate((menos_W, [R]))
                return R

            # Caso 3,  R < N --> contraccion
            if R_ < N_:
                self.last_vertex = self.x_vertex_data
                self.last_response = self.y_vertex_data
                CR = 1.5 * M - 0.5 * W
                self.last_point_added = CR
                self.last_point_added_type = "CR"

                self.current_vertex = np.concatenate((menos_W, [CR]))
                return CR
            ##### Continuar mas adelante y pensar mejor mec de control.
