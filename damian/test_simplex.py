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
    model = SimplexOpt(bounds=bounds, step=1)
    print(model.current_vertex)
    model.simplex(model.current_vertex, [1, 2, 3])
    print(model.current_vertex)
