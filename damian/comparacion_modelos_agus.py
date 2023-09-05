import numpy as np
import pandas as pd
from bayopt import BayOptRBF
import itertools

# X = [Independiente,	pH,	Carga,	Conc,	pH*carga,	pH*Conc,	Carga*Conc,	pH^2,	Carga^2,	Conc^2]

mn = np.array(
    [
        31448870289,
        -554525034.7,
        -14265048609,
        -1935511006,
        156735633,
        -20701752.16,
        116064658.9,
        293386201.6,
        1780377838,
        237933042.5,
    ]
)

sb = np.array(
    [
        80734.67046,
        370.8011381,
        -16258.72842,
        5740.418968,
        137.9395184,
        -293.5922591,
        -8.785053956,
        71.05908436,
        1509.728314,
        -214.2989442,
    ]
)

cd = np.array(
    [
        75814864409,
        9577014119,
        -39742172161,
        -7293630169,
        196753959.9,
        -534409051.3,
        731416776.9,
        157535411.9,
        4577912599,
        638701385.6,
    ]
)

pb = np.array(
    [
        456292.8511,
        112339.4885,
        -64596.93522,
        51866.41538,
        7300.303724,
        -1713.886322,
        407.9571991,
        -9415.365197,
        643.7838491,
        -2283.865801,
    ]
)
exp_mn = 1 / 1.78
exp_sb = 1
exp_cd = 1 / 2.02
exp_pb = 1


# X = [ph, carga, concentracion]
def respuesta(X, coef_elemento, exponente):
    X_poly = [
        X[0],
        X[1],
        X[2],
        X[0] * X[1],
        X[0] * X[2],
        X[1] * X[2],
        X[0] ** 2,
        X[1] ** 2,
        X[2] ** 2,
    ]
    return (coef_elemento[0] + np.dot(X_poly, coef_elemento[1:])) ** exponente


# test = [1, 1, 1]
# print(respuesta(test, mn, exp_mn))

# VARS
n_dim = 3
# ph-carga-concentracion
bounds_min = [3, 1, 1]
bounds_max = [9, 5, 10]
bounds = np.array([*zip(bounds_min, bounds_max)])
# datos iniciales
x_data = np.array(list(itertools.product(*zip(bounds_min, bounds_max))))
x_data = np.concatenate((x_data, np.zeros(shape=(1, n_dim))))
y_data = np.array([respuesta(x, sb, exp_sb) for x in x_data])
# modelo
model = BayOptRBF(x_data, y_data)
# optimizacion

pct_max = []
tested_values = []
# cd, mn, pb, sb
x_max_agustin = [[9, 1, 1], [8.8, 1, 10], [5.5, 1, 10], [3.5, 1, 10]]
y_max_agustin = [
    respuesta(x, params, exp)
    for x, params, exp in zip(
        x_max_agustin, [cd, mn, pb, sb], [exp_cd, exp_mn, exp_pb, exp_sb]
    )
]

for iteration in range(10):
    x_next, acq_func_val = model.global_max(acq_fun="EI", bounds=bounds)
    # valor de la funcion en el punto encontrado
    # como la funcion esta normalizada entre 0 y 1, tmb es la fraccion del maximo.
    y_next = respuesta(x_next, sb, exp_sb)
    pct_max.append(y_next)
    tested_values.append(x_next)
    # agrego el nuevo punto a los datos para entrenar el modelo
    x_data = np.concatenate((x_data, [x_next]))
    y_data = np.concatenate((y_data, [y_next]))
    # fiteo el model
    model.fit(x_data, y_data)

print(pct_max / y_max_agustin[3])
print(tested_values)
