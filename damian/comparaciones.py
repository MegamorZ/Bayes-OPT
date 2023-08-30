import numpy as np
from bayopt import BayOptRBF, RandomQuadratic, SimplexOpt, RandomGaussian
import warnings
import itertools
import pandas as pd

warnings.filterwarnings("ignore")


if __name__ == "__main__":
    # vars
    n_dim = 6

    _1d_bound = [-1, 1]

    bounds = np.array([_1d_bound for _ in range(n_dim)])

    N = 50  # numero de experimentos

    # listas para almacenar los resultados de las optimizaciones para cada superficie.
    data_bayopt = []
    data_simplex = []

    for exp in range(N):
        # inicializo superficie
        RS = RandomQuadratic(n_dim=n_dim, bounds=bounds, offset=True, noise=0.01)

        # datos iniciales para bayopt (centro + bordes)
        x_data_bayopt = np.array(list(itertools.product(_1d_bound, repeat=n_dim)))
        x_data_bayopt = np.concatenate((x_data_bayopt, np.zeros(shape=(1, n_dim))))
        y_data_bayopt = np.array([RS(x) for x in x_data_bayopt])

        ## modelos
        model_1 = BayOptRBF(x_data_bayopt, y_data_bayopt)
        model_2 = SimplexOpt(bounds=bounds, step=0.1)

        ## optimizacion

        pct_max_model_bayopt = []
        pct_max_model_simplex = []
        curr_max_bayot = 0
        curr_max_simplex = 0

        # optimizo los siguientes 10 puntos
        for iteration in range(10):
            ##Modelo BayOptRBF
            x_next, acq_func_val = model_1.global_max(acq_fun="EI", bounds=bounds)
            # valor de la funcion en el punto encontrado
            # como la funcion esta normalizada entre 0 y 1, tmb es la fraccion del maximo.
            y_next = RS(x_next)

            # comparo el nuevo punto con el mejor maximo hasta el momento
            if y_next > curr_max_bayot:
                curr_max_bayot = y_next

            pct_max_model_bayopt.append(curr_max_bayot)

            # agrego el nuevo punto a los datos para entrenar el modelo
            x_data_bayopt = np.concatenate((x_data_bayopt, [x_next]))
            y_data_bayopt = np.concatenate((y_data_bayopt, [y_next]))
            # fiteo el model
            model_1.fit(x_data_bayopt, y_data_bayopt)

            ##Modelo Simplex

            x_vertex_data = model_2.current_vertex
            y_vertex_data = [RS(x) for x in x_vertex_data]
            # Calculo el proximo punto en funcion del vertex actual.
            x_next = model_2.simplex(x_vertex_data, y_vertex_data)
            y_next = RS(x_next)

            # comparo el nuevo punto con el mejor maximo hasta el momento
            if y_next > curr_max_simplex:
                curr_max_simplex = y_next

            pct_max_model_simplex.append(curr_max_simplex)

        # Guardo la optimizacion para esta superficie
        data_bayopt.append(pct_max_model_bayopt)
        data_simplex.append(pct_max_model_simplex)

    data_bayopt_mean = np.array(data_bayopt).mean(axis=0)
    data_bayopt_sd = np.array(data_bayopt).std(axis=0)
    data_simplex_mean = np.array(data_simplex).mean(axis=0)
    data_simplex_sd = np.array(data_simplex).std(axis=0)

    pd.DataFrame(
        {
            "bayopt_mean": data_bayopt_mean,
            "bayopt_sd": data_bayopt_sd,
            "simplex_mean": data_simplex_mean,
            "simplex_sd": data_simplex_sd,
        }
    ).to_csv(f"datos_ndim{n_dim}")
