import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
from bayopt import BayOptRBF, RandomQuadratic, RandomGaussian

if __name__ == '__main__':

    n_dim=2
    RS = RandomGaussian(n_dim=n_dim, offset=False)

    #x_data = np.random.uniform(low=-2, high=2, size=(10, n_dim))
    x_data = np.array([[0,0],[2,2],[2,-2],[-2,2],[-2,-2]])
    #x_data = np.array([[0,0],[1,1],[1,-1],[-1,1],[-1,-1],[2,2],[2,-2],[-2,2],[-2,-2]])
    y_data = np.array([RS(x) for x in x_data])
    bounds = np.array([(-2,2),(-2,2)])

    model = BayOptRBF(x_data,y_data)

    print(x_data)
    print(y_data)
    print(model.regressor.kernel_.get_params())

    # figura
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    # mesh
    x = y = np.linspace(-3, 3, 20)
    X, Y = np.meshgrid(x, y)
    # convierto a coordenadas y calculo el valor en cada punto
    z = np.array([])
    for x0, y0 in zip(np.ravel(X), np.ravel(Y)):
        z = np.append(z, model.regressor.predict(np.array([[x0, y0]])))
    # convierto los valores a forma de mesh y ploteo
    Z = z.reshape(X.shape)

    ax.plot_surface(X, Y, Z)

    ax.set_xlabel("X Label")
    ax.set_ylabel("Y Label")
    ax.set_zlabel("Z Label")

    plt.show()
