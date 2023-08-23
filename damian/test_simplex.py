import numpy as np
from bayopt import SimplexOpt, RandomQuadratic
import itertools

if __name__ == "__main__":
    # vars
    n_dim = 2

    _1d_bound = [-1, 1]

    bounds = np.array([_1d_bound for _ in range(n_dim)])

    # data
    x_data = np.array(list(itertools.product(_1d_bound, repeat=n_dim)))
    x_data = np.concatenate((x_data, np.zeros(shape=(1, n_dim))))

    RS = RandomQuadratic(n_dim=n_dim, bounds=bounds, offset=True, noise=0.01)
    y_data = np.array([RS(x) for x in x_data])

    # model
    # print(x_data, y_data)
    model = SimplexOpt(bounds=bounds, step=0.1)

    # opt

    pct_max = []
    tested_values = []
    for iteration in range(20):
        x_vertex_data = model.current_vertex
        y_vertex_data = [RS(x) for x in x_vertex_data]

        x_next = model.simplex(x_vertex_data, y_vertex_data)
        y_next = RS(x_next)

        pct_max.append(y_next)
        tested_values.append(x_next)

    for index, value in enumerate(pct_max):
        print(f"n={index}, pct_max={100*value}, X= {tested_values[index]}")
