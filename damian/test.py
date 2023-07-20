import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from bayopt import BayOptRBF, RandomQuadratic

if __name__ == '__main__':
    #generando una funcion cuadratica
    func = RandomQuadratic(n_dim=2)
    #bounds
    bounds=[(-2,2),(-2,2)]

    #ploteando
    densidad_meshgrid = 5 #puntos
    x_mesh= np.linspace(-2,2,densidad_meshgrid)
    y_mesh= np.linspace(-2,2,densidad_meshgrid)
    X,Y = np.meshgrid(x_mesh,y_mesh, sparse=True)

    print(X,Y)
    #Z lista de lista con los valores de la funcion en cada punto del grid
    xs = X[0]
    ys = Y.reshape(1,densidad_meshgrid)[0]

    Z = np.array([[func(np.array([[x,y]])) for x in xs] for y in ys])
    
    #FIG SURFACE
    fig, ax = plt.subplots(subplot_kw={"projection": "3d"})

    # Plot the surface.
    surf = ax.plot_surface(X, Y, Z, cmap=cm.coolwarm,
                        linewidth=0, antialiased=False)

    plt.show()
