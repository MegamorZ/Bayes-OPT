import numpy as np
from bayopt import BayOptRBF, RandomQuadratic, RandomGaussian, plot_3d, bounded_min

if __name__ == "__main__":
    # np.random.seed(42)

    # Inicializo los datos.
    n_dim = 2
    bounds = np.array([(-1, 1), (-1, 1)])
    RS = RandomQuadratic(n_dim=n_dim, bounds=bounds, offset=True, noise=0.01)

    x_data = np.array([[0, 0], [1, 1], [1, -1], [-1, 1], [-1, -1]])
    # x_data = np.random.uniform(low=-2, high=2, size=(500, n_dim))
    # x_data = np.array([[0,0],[1,1],[1,-1],[-1,1],[-1,-1],[2,2],[2,-2],[-2,2],[-2,-2]])
    y_data = np.array([RS(x) for x in x_data])

    model = BayOptRBF(x_data, y_data)

    plot_3d(RS, model.regressor)
