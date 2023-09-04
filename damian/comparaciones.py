import numpy as np
from doepy import build
from bayopt import BayOptRBF, RandomQuadratic, SimplexOpt, RandomGaussian
import warnings
import itertools
import pandas as pd
from sklearn.preprocessing import PolynomialFeatures
from sklearn import linear_model

warnings.filterwarnings("ignore")


if __name__ == "__main__":
    for n_dim in [2, 3, 4, 5]:
        # vars
        _1d_bound = [-1, 1]
        bounds = np.array([_1d_bound for _ in range(n_dim)])
        N = 50  # numero de experimentos

        # listas para almacenar los resultados de las optimizaciones para cada superficie.
        data_bayopt_ei = []
        data_simplex = []
        data_bayopt_pi = []
        data_bayopt_ucb = []

        best_per_surface_bayopt_ei = []
        best_per_surface_bayopt_pi = []
        best_per_surface_bayopt_ucb = []
        best_per_surface_simplex = []
        data_ccd = []

        for exp in range(N):
            # inicializo superficie
            RS = RandomQuadratic(n_dim=n_dim, bounds=bounds, offset=True, noise=0.02)

            # datos iniciales para bayopt (centro + bordes)
            x_data_bayopt = np.array(list(itertools.product(_1d_bound, repeat=n_dim)))
            x_data_bayopt = np.concatenate((x_data_bayopt, np.zeros(shape=(1, n_dim))))
            y_data_bayopt = np.array([RS(x) for x in x_data_bayopt])

            ## modelos
            model_1 = BayOptRBF(x_data_bayopt, y_data_bayopt)  # EI
            model_2 = SimplexOpt(bounds=bounds, step=0.1)
            model_3 = BayOptRBF(x_data_bayopt, y_data_bayopt)  # PI
            model_4 = BayOptRBF(x_data_bayopt, y_data_bayopt)  # UCB

            ## optimizacion

            pct_max_model_bayopt_ei = []
            pct_max_model_bayopt_pi = []
            pct_max_model_bayopt_ucb = []
            pct_max_model_simplex = []

            curr_max_bayot_ei = 0
            curr_max_bayot_pi = 0
            curr_max_bayot_ucb = 0
            curr_max_simplex = 0

            # optimizo cada modelo
            ##Modelo BayOptRBF con EI como func de adquisicion
            for iteration in range(10):
                x_next, acq_func_val = model_1.global_max(acq_fun="EI", bounds=bounds)
                # valor de la funcion en el punto encontrado
                # como la funcion esta normalizada entre 0 y 1, tmb es la fraccion del maximo.
                y_next = RS(x_next)

                # comparo el nuevo punto con el mejor maximo hasta el momento
                if y_next > curr_max_bayot_ei:
                    curr_max_bayot_ei = y_next

                pct_max_model_bayopt_ei.append(curr_max_bayot_ei)

                # agrego el nuevo punto a los datos para entrenar el modelo
                x_data_bayopt = np.concatenate((x_data_bayopt, [x_next]))
                y_data_bayopt = np.concatenate((y_data_bayopt, [y_next]))
                # fiteo el model
                model_1.fit(x_data_bayopt, y_data_bayopt)

            ##Modelo Simplex
            for iteration in range(
                10 + 2**n_dim - n_dim
            ):  # para que haya la misma cantidad de puntos que en modelo bayesiano.
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

            ## CCD + ajuste cuadratico para la superficie
            ccd_model = build.central_composite(
                {str(dim): [-1, 1] for dim in range(n_dim)}, face="cci"
            )
            # genero los datos
            x_data_ccd = ccd_model.to_numpy()
            y_data_ccd = np.array([RS(x) for x in x_data_ccd])

            # ajusto un modelo polinomico
            poly = PolynomialFeatures(degree=2)
            x_poly = poly.fit_transform(x_data_ccd)

            poly_model = linear_model.LinearRegression()
            poly_model.fit(x_poly, y_data_ccd)
            # busco el maximo
            y_pred = poly_model.predict(x_poly)

            generator = np.meshgrid(*(np.linspace(-1, 1, 10) for dim in range(n_dim)))
            x_search = np.array([x.flatten() for x in generator]).T
            x_search = poly.transform(x_search)

            y_pred = poly_model.predict(x_search)
            x_optimal = x_search[np.argmax(y_pred)][
                1 : n_dim + 1
            ]  # en coordenadas lineales
            data_ccd.append(RS(x_optimal))

            ##Modelo BayOptRBF con PI como func de adquisicion
            # datos iniciales para bayopt (centro + bordes)
            x_data_bayopt = np.array(list(itertools.product(_1d_bound, repeat=n_dim)))
            x_data_bayopt = np.concatenate((x_data_bayopt, np.zeros(shape=(1, n_dim))))
            y_data_bayopt = np.array([RS(x) for x in x_data_bayopt])

            for iteration in range(10):
                x_next, acq_func_val = model_3.global_max(acq_fun="PI", bounds=bounds)
                # valor de la funcion en el punto encontrado
                # como la funcion esta normalizada entre 0 y 1, tmb es la fraccion del maximo.
                y_next = RS(x_next)

                # comparo el nuevo punto con el mejor maximo hasta el momento
                if y_next > curr_max_bayot_pi:
                    curr_max_bayot_pi = y_next

                pct_max_model_bayopt_pi.append(curr_max_bayot_pi)

                # agrego el nuevo punto a los datos para entrenar el modelo
                x_data_bayopt = np.concatenate((x_data_bayopt, [x_next]))
                y_data_bayopt = np.concatenate((y_data_bayopt, [y_next]))
                # fiteo el model
                model_3.fit(x_data_bayopt, y_data_bayopt)

            ##Modelo BayOptRBF con UCB como func de adquisicion
            # datos iniciales para bayopt (centro + bordes)
            x_data_bayopt = np.array(list(itertools.product(_1d_bound, repeat=n_dim)))
            x_data_bayopt = np.concatenate((x_data_bayopt, np.zeros(shape=(1, n_dim))))
            y_data_bayopt = np.array([RS(x) for x in x_data_bayopt])

            for iteration in range(10):
                x_next, acq_func_val = model_4.global_max(acq_fun="UCB", bounds=bounds)
                # valor de la funcion en el punto encontrado
                # como la funcion esta normalizada entre 0 y 1, tmb es la fraccion del maximo.
                y_next = RS(x_next)

                # comparo el nuevo punto con el mejor maximo hasta el momento
                if y_next > curr_max_bayot_ucb:
                    curr_max_bayot_ucb = y_next

                pct_max_model_bayopt_ucb.append(curr_max_bayot_ucb)

                # agrego el nuevo punto a los datos para entrenar el modelo
                x_data_bayopt = np.concatenate((x_data_bayopt, [x_next]))
                y_data_bayopt = np.concatenate((y_data_bayopt, [y_next]))
                # fiteo el model
                model_4.fit(x_data_bayopt, y_data_bayopt)

            # Guardo la optimizacion para esta superficie
            data_bayopt_ei.append(pct_max_model_bayopt_ei)
            data_bayopt_pi.append(pct_max_model_bayopt_pi)
            data_bayopt_ucb.append(pct_max_model_bayopt_ucb)
            data_simplex.append(pct_max_model_simplex)
            best_per_surface_bayopt_ei.append(pct_max_model_bayopt_ei[-1])
            best_per_surface_bayopt_pi.append(pct_max_model_bayopt_pi[-1])
            best_per_surface_bayopt_ucb.append(pct_max_model_bayopt_ucb[-1])
            best_per_surface_simplex.append(pct_max_model_simplex[-1])

        data_bayopt_mean_ei = np.array(data_bayopt_ei).mean(axis=0)
        data_bayopt_sd_ei = np.array(data_bayopt_ei).std(axis=0)
        best_per_surface_bayopt_ei = np.array(best_per_surface_bayopt_ei)

        data_bayopt_mean_pi = np.array(data_bayopt_pi).mean(axis=0)
        data_bayopt_sd_pi = np.array(data_bayopt_pi).std(axis=0)
        best_per_surface_bayopt_pi = np.array(best_per_surface_bayopt_pi)

        data_bayopt_mean_ucb = np.array(data_bayopt_ucb).mean(axis=0)
        data_bayopt_sd_ucb = np.array(data_bayopt_ucb).std(axis=0)
        best_per_surface_bayopt_ucb = np.array(best_per_surface_bayopt_ucb)

        data_simplex_mean = np.array(data_simplex).mean(axis=0)[-10:]
        data_simplex_sd = np.array(data_simplex).std(axis=0)[-10:]
        best_per_surface_simplex = np.array(best_per_surface_simplex)

        pd.DataFrame(
            {
                "num": np.arange(start=1 + 2**n_dim, stop=11 + 2**n_dim, step=1),
                "bayopt_mean_ei": data_bayopt_mean_ei,
                "bayopt_sd_ei": data_bayopt_sd_ei,
                "bayopt_mean_pi": data_bayopt_mean_pi,
                "bayopt_sd_pi": data_bayopt_sd_pi,
                "bayopt_mean_ucb": data_bayopt_mean_ucb,
                "bayopt_sd_ucb": data_bayopt_sd_ucb,
                "simplex_mean": data_simplex_mean,
                "simplex_sd": data_simplex_sd,
            }
        ).to_csv(f"datos_resumidos_ndim{n_dim}")

        pd.DataFrame(
            {
                "bayopt_best_ei": best_per_surface_bayopt_ei,
                "bayopt_best_pi": best_per_surface_bayopt_pi,
                "bayopt_best_ucb": best_per_surface_bayopt_ucb,
                "simplex_best": best_per_surface_simplex,
            }
        ).to_csv(f"datos_optimos_por_superficie_ndim{n_dim}")

        pd.DataFrame(
            {
                "num": np.array([2**n_dim + n_dim + 5 for _ in range(len(data_ccd))]),
                "ccd_best": np.array(data_ccd),
            }
        ).to_csv(f"datos_ccd_ndim{n_dim}")
