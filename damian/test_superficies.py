import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
from bayopt import BayOptRBF, RandomQuadratic, RandomGaussian

if __name__ == "__main__":
    # generando una funcion cuadratica
    #func = RandomQuadratic(n_dim=2)
    func = RandomGaussian(n_dim=2)
    # bounds
    bounds = [(-2, 2), (-2, 2)]

    # figura
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    # mesh
    x = y = np.linspace(-3, 3, 20)
    X, Y = np.meshgrid(x, y)
    # convierto a coordenadas y calculo el valor en cada punto
    z = np.array([])
    for x0, y0 in zip(np.ravel(X), np.ravel(Y)):
        z = np.append(z, func(np.array([[x0, y0]])))
    # convierto los valores a forma de mesh y ploteo
    Z = z.reshape(X.shape)

    ax.plot_surface(X, Y, Z)

    ax.set_xlabel("X Label")
    ax.set_ylabel("Y Label")
    ax.set_zlabel("Z Label")
    print(f"x0:{func.x0},c:{func.c}")

    plt.show()
