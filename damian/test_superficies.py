import numpy as np
from bayopt import BayOptRBF, RandomQuadratic, RandomGaussian, plot_3d, bounded_min

if __name__ == "__main__":
    # np.random.seed(42)

    # Inicializo los datos.
    n_dim = 2
    bounds = np.array([(-1, 1), (-1, 1)])
    # RS = RandomQuadratic(n_dim=n_dim, bounds=bounds, offset=True, noise=0.01)
    RS = RandomGaussian(n_dim=n_dim, offset=True, noise=0.01)

    # x_data = np.array([[0, 0], [1, 1], [1, -1], [-1, 1], [-1, -1]])
    x_data = np.random.uniform(low=-1, high=1, size=(100, n_dim))

    y_data = np.array([RS(x) for x in x_data])

    model = BayOptRBF(x_data, y_data)

    print(model.kernel)
    plot_3d(RS, model.regressor)
