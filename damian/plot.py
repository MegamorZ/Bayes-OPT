import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def plot_opt(num, mean, sd, legend, n_dim):
    plt.figure(figsize=(12, 10), dpi=80)
    plt.plot(num, mean, "r", linewidth=4, label=legend)
    plt.fill_between(
        num,
        mean - 2 * sd,
        mean + 2 * sd,
        alpha=0.2,
        fc="b",
        ec="None",
        label="95% confidence interval",
    )
    plt.fill_between(
        num,
        mean - 1.5 * sd,
        mean + 1.5 * sd,
        alpha=0.2,
        fc="b",
        ec="None",
    )  # , label='95% confidence interval')
    plt.fill_between(
        num,
        mean - 1 * sd,
        mean + 1 * sd,
        alpha=0.2,
        fc="b",
        ec="None",
    )  # , label='95% confidence interval')
    plt.fill_between(
        num,
        mean - 0.5 * sd,
        mean + 0.5 * sd,
        alpha=0.2,
        fc="b",
        ec="None",
    )  # , label='95% confidence interval')
    plt.xlabel("Number of Measurements", fontsize=24)
    plt.ylabel("% of optimum value achieved", fontsize=24)
    plt.xlim(num.min(), num.max())
    plt.ylim((mean - 2 * sd).min() - 0.05, 1.001)
    plt.title(f"{n_dim} Factores", fontsize=24)
    plt.legend(loc="lower right", prop={"size": 20})
    plt.show()


n_dim = 3
data = pd.read_csv(f"datos_resumidos_ndim{n_dim}")
data_ccd = pd.read_csv(f"datos_ccd_ndim{n_dim}")

print(data_ccd.ccd_best.mean())
# ver como corrijo valores ligeramente por sobre 1 debido al ruido.add()
data.bayopt_mean[data.bayopt_mean > 1] = 1

#
plot_opt(
    data.num,
    data.bayopt_mean,
    data.bayopt_sd,
    legend="Mean Bayesian Optimization",
    n_dim=n_dim,
)

plot_opt(
    data.num,
    data.simplex_mean,
    data.simplex_sd,
    legend="Mean Simplex Optimization",
    n_dim=n_dim,
)
