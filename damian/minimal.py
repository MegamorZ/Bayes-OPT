from typing import Any
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, Matern,RationalQuadratic
from scipy.optimize import minimize
from scipy.spatial.distance import pdist
from scipy.stats import norm
from bayopt import RandomGaussian, RandomQuadratic

#datos

x_data_0 = np.array([[0,0],[2,2],[2,-2],[-2,2],[-2,-2]])
x_data_1 = np.random.uniform(low=-2, high=2, size=(1,2))
x_data = np.concatenate([x_data_0, x_data_1]) 

RS = RandomQuadratic(n_dim=2, noise=0.1)
#RS = RandomGaussian(n_dim=2)
y_data = np.array([RS(x) for x in x_data])

#lengthscale bounds
_dist = pdist(x_data)
_dist_max = np.max(_dist)
_dist_min = 2*np.min(_dist) if np.min(_dist) > 0.5 else 0.5

#kernel
#kernel = Matern(length_scale_bounds=(_dist_min, _dist_max))
#kernel = RationalQuadratic(length_scale_bounds=(_dist_min, _dist_max))
kernel = RBF(length_scale_bounds=(_dist_min, _dist_max))
gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=9, alpha=0.1, normalize_y=True)
gp.fit(x_data,y_data)

# #debug
# print(gp.kernel_.get_params())
# print(f"XDATA {x_data}")
# print(f"YDATA {y_data}")

#plot function
# figura
fig, ax = plt.subplots(1,2,figsize=(10,10),subplot_kw=dict(projection='3d'))

# mesh
x = y = np.linspace(-3, 3, 20)
X, Y = np.meshgrid(x, y)
# convierto a coordenadas y calculo el valor en cada punto
z = np.array([])
z_aprox =np.array([])
for x0, y0 in zip(np.ravel(X), np.ravel(Y)):
    z = np.append(z, RS(np.array([[x0, y0]])))
    z_aprox = np.append(z_aprox, gp.predict(np.array([[x0, y0]])))

# convierto los valores a forma de mesh y ploteo
Z = z.reshape(X.shape)
Z_aprox = z_aprox.reshape(X.shape)

ax[0].plot_surface(X, Y, Z)

ax[1].plot_surface(X, Y, Z_aprox)

ax[0].set_xlabel("X Label")
ax[0].set_ylabel("Y Label")
ax[0].set_zlabel("Z Label")

ax[1].set_xlabel("X Label")
ax[1].set_ylabel("Y Label")
ax[1].set_zlabel("Z Label")

ax[0].set_title("Ground Truth")
ax[1].set_title("Aproximation")
plt.show()


### Mismo comportamiento que usando la clase.