import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


def UCB(x, gp, kappa=1.96):
    # Calcular media y desvio para el valor de prueba "x"
    mean, std = gp.predict([x], return_std=True)

    # calcular y devolver la probabilidad de mejora
    result = mean + kappa * std
    return result


X = np.linspace(start=0, stop=10, num=1_000).reshape(-1, 1)
y = np.squeeze(X * np.sin(X))

rng = np.random.RandomState(1)
# training_indices = rng.choice(np.arange(y.size), size=6, replace=False)
training_indices = [20, 920, 606]
training_indices = [20, 920, 606, 999]
training_indices = [20, 920, 606, 999, 800]
training_indices = [20, 920, 606, 999, 800, 275]
X_train, y_train = X[training_indices], y[training_indices]

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF

kernel = 1 * RBF(length_scale=1.0, length_scale_bounds=(1, 1e2))
gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=9)
gp.fit(X_train, y_train)
gp.kernel_

mean_prediction, std_prediction = gp.predict(X, return_std=True)
# creo plot
fig, axes = plt.subplots(2, 1, gridspec_kw={"height_ratios": [3, 1]}, figsize=(6, 5))
# subplot0
axes[0].plot(X, y, label=r"$f(x) = x \sin(x)$", linestyle="dashed", c="#2A2E45")
axes[0].scatter(X_train, y_train, label="Observaciones", c="#9A031E")
axes[0].plot(X, mean_prediction, label="Predicción", c="#9A031E")
axes[0].fill_between(
    X.ravel(),
    mean_prediction - 1.96 * std_prediction,
    mean_prediction + 1.96 * std_prediction,
    alpha=0.5,
    label=r"Intervalo de confianza 95% ",
    fc="#FB8B24",
)
axes[0].legend(loc="upper left")
axes[0].set_ylabel("$f(x)$")
axes[0].set_ylim(bottom=-12, top=14)

# subplot1
y_ucb = np.array([UCB(x, gp) for x in X])
axes[1].plot(X, y_ucb, c="#9A031E", label="Funcion de adquisición UCB")

max_index = y_ucb.argmax()
axes[1].vlines(X[max_index], y_ucb.min(), y_ucb.max(), linestyles="dotted", color="r")
axes[1].scatter(
    X[max_index],
    y_ucb[max_index],
    marker="*",
    s=150,
    c="#FB8B24",
    edgecolors="#2A2E45",
    label="Maximo de la funcion de adquisición",
)

# axes[1].legend(loc="lower left")
axes[1].set_xlabel("$x$")
axes[1].set_ylabel("$UCB(x)$")


# fig.suptitle("")
plt.tight_layout()
plt.savefig(f"bayopt_example{len(training_indices)}", dpi=300)
plt.show()
