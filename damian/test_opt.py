import numpy as np
from bayopt import BayOptRBF, RandomQuadratic
import warnings
import itertools

warnings.filterwarnings("ignore")


if __name__ == "__main__":
    # vars
    n_dim = 10

    _1d_bound = [-1, 1]

    bounds = np.array([_1d_bound for _ in range(n_dim)])

    # data
    x_data = np.array(list(itertools.product(_1d_bound, repeat=n_dim)))
    x_data = np.concatenate((x_data, np.zeros(shape=(1, n_dim))))

    RS = RandomQuadratic(n_dim=n_dim, bounds=bounds, offset=True, noise=0.01)
    y_data = np.array([RS(x) for x in x_data])

    # model
    # print(x_data, y_data)
    model = BayOptRBF(x_data, y_data)

    # optimizacion
    pct_max = []
    tested_values = []
    for iteration in range(10):
        x_next, acq_func_val = model.global_max(acq_fun="EI", bounds=bounds)
        # valor de la funcion en el punto encontrado
        # como la funcion esta normalizada entre 0 y 1, tmb es la fraccion del maximo.
        y_next = RS(x_next)
        pct_max.append(y_next)
        tested_values.append(x_next)
        # agrego el nuevo punto a los datos para entrenar el modelo
        x_data = np.concatenate((x_data, [x_next]))
        y_data = np.concatenate((y_data, [y_next]))
        # fiteo el model
        model.fit(x_data, y_data)

    for index, value in enumerate(pct_max):
        print(f"n={index}, pct_max={100*value}, X= {tested_values[index]}")
