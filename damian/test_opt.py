import numpy as np
from bayopt import BayOptRBF, RandomQuadratic
import warnings

warnings.filterwarnings("ignore")


if __name__ == "__main__":
    # vars
    n_dim = 2
    bounds = np.array([(-1, 1), (-1, 1)])

    # data
    x_data = np.array([[0, 0], [1, 1], [1, -1], [-1, 1], [-1, -1]])

    RS = RandomQuadratic(n_dim=n_dim, bounds=bounds, offset=True, noise=False)
    y_data = np.array([RS(x) for x in x_data])

    # model
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
