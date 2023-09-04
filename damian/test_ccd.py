from doepy import build
import numpy as np
from bayopt import RandomQuadratic, bounded_max
import itertools

if __name__ == "__main__":
    # vars
    n_dim = 2

    _1d_bound = [-1, 1]
    bounds = np.array([_1d_bound for _ in range(n_dim)])
    # inicializo superficie
    RS = RandomQuadratic(n_dim=n_dim, bounds=bounds, offset=True, noise=0.01)

    # armo el modelo
    ccd_model = build.central_composite(
        {str(dim): [-1, 1] for dim in range(n_dim)}, face="cci"
    )
    # genero los datos
    x_data = ccd_model.to_numpy()
    y_data = np.array([RS(x) for x in x_data])

    # ajusto un modelo polinomico

    from sklearn.preprocessing import PolynomialFeatures
    from sklearn import linear_model

    poly = PolynomialFeatures(degree=2)
    x_poly = poly.fit_transform(x_data)

    poly_model = linear_model.LinearRegression()
    poly_model.fit(x_poly, y_data)

    y_pred = poly_model.predict(x_poly)

    generator = np.meshgrid(*(np.linspace(-1, 1, 10) for dim in range(n_dim)))
    x_search = np.array([x.flatten() for x in generator]).T
    x_search = poly.transform(x_search)

    y_pred = poly_model.predict(x_search)
    x_optimal = x_search[np.argmax(y_pred)][1 : n_dim + 1]  # en coordenadas lineales
    y_optimal = RS(x_optimal)

    print(x_optimal, RS.x0)
    print(y_optimal)
    print(x_data.shape)
