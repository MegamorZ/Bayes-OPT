import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


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


n_dim = 5
data = pd.read_csv(f"datos_resumidos_ndim{n_dim}")
data_ccd = pd.read_csv(f"datos_ccd_ndim{n_dim}")
data_violin = pd.read_csv(f"datos_optimos_por_superficie_ndim{n_dim}")
data_violin["ccd_best"] = data_ccd["ccd_best"]
data_violin.drop(["Unnamed: 0"], axis=1, inplace=True)

print(data_violin.columns)

# ver como corrijo valores ligeramente por sobre 1 debido al ruido.add()
data.bayopt_mean_ei[data.bayopt_mean_ei > 1] = 1
# for column_name, column_data in data_violin.iteritems():
#     if column_data.max() > 1:
#         data_violin[column_name] = data_violin[column_name] + 1 - column_data.max()
if n_dim < 5:
    data_violin.bayopt_best_ei = (
        data_violin.bayopt_best_ei + 1 - data_violin.bayopt_best_ei.mean()
    )
    data_violin.bayopt_best_pi = (
        data_violin.bayopt_best_pi + 1 - data_violin.bayopt_best_pi.mean()
    )
    data_violin.bayopt_best_ucb = (
        data_violin.bayopt_best_ucb + 1 - data_violin.bayopt_best_ucb.mean()
    )


##creo subplots
fig, axes = plt.subplots(
    3, 1, figsize=(6, 10.4)
)  # configurar figsize antes de usar en poster (pulgadas)
# TOP, optimizacion simplex
axes[0].plot(
    data.num,
    data.bayopt_mean_ei,
    "#9A031E",
    linewidth=3,
    label="Promedio de la optimizacion Bayesiana.",
)
axes[0].fill_between(
    data.num,
    data.bayopt_mean_ei - 2 * data.bayopt_sd_ei,
    data.bayopt_mean_ei + 2 * data.bayopt_sd_ei,
    alpha=0.15,
    fc="#FB8B24",
    ec="None",
    label="Intervalo de confianza 95%",
)
axes[0].fill_between(
    data.num,
    data.bayopt_mean_ei - 1.5 * data.bayopt_sd_ei,
    data.bayopt_mean_ei + 1.5 * data.bayopt_sd_ei,
    alpha=0.15,
    fc="#FB8B24",
    ec="None",
)
axes[0].fill_between(
    data.num,
    data.bayopt_mean_ei - 1 * data.bayopt_sd_ei,
    data.bayopt_mean_ei + 1 * data.bayopt_sd_ei,
    alpha=0.15,
    fc="#FB8B24",
    ec="None",
)
axes[0].fill_between(
    data.num,
    data.bayopt_mean_ei - 0.5 * data.bayopt_sd_ei,
    data.bayopt_mean_ei + 0.5 * data.bayopt_sd_ei,
    alpha=0.15,
    fc="#FB8B24",
    ec="None",
)
# limites
axes[0].set_xlim(data.num.min(), data.num.max())
axes[0].set_ylim((data.bayopt_mean_ei - 2 * data.bayopt_sd_ei).min() - 0.05, 1.005)
axes[0].legend(loc="lower right", prop={"size": 9})
axes[0].set_ylabel("Fraccion del maximo")
axes[0].set_xlabel("Mediciones")

##Middle, optimizacion simplex
axes[1].plot(
    data.num,
    data.simplex_mean,
    "#9A031E",
    linewidth=3,
    label="Promedio de la optimizacion Simplex.",
)
axes[1].fill_between(
    data.num,
    data.simplex_mean - 2 * data.simplex_sd,
    data.simplex_mean + 2 * data.simplex_sd,
    alpha=0.15,
    fc="#FB8B24",
    ec="None",
    label="Intervalo de confianza 95%",
)
axes[1].fill_between(
    data.num,
    data.simplex_mean - 1.5 * data.simplex_sd,
    data.simplex_mean + 1.5 * data.simplex_sd,
    alpha=0.15,
    fc="#FB8B24",
    ec="None",
)
axes[1].fill_between(
    data.num,
    data.simplex_mean - 1 * data.simplex_sd,
    data.simplex_mean + 1 * data.simplex_sd,
    alpha=0.15,
    fc="#FB8B24",
    ec="None",
)
axes[1].fill_between(
    data.num,
    data.simplex_mean - 0.5 * data.simplex_sd,
    data.simplex_mean + 0.5 * data.simplex_sd,
    alpha=0.15,
    fc="#FB8B24",
    ec="None",
)

# limites
axes[1].set_xlim(data.num.min(), data.num.max())
axes[1].set_ylim((data.simplex_mean - 2 * data.simplex_sd).min() - 0.15, 1.002)
axes[1].legend(loc="lower right", prop={"size": 9})
axes[1].set_ylabel("Fraccion del maximo")
axes[1].set_xlabel("Mediciones")

##Violines

v_plot = axes[2].violinplot(
    data_violin[
        [
            "bayopt_best_ei",
            "bayopt_best_pi",
            "bayopt_best_ucb",
            "simplex_best",
            "ccd_best",
        ]
    ],
    showextrema=True,
    showmedians=True,
)
# cambio los colores de los violines
colors = ["#5F0F40", "#9A031E", "#FB8B24", "#E36414", "#2A2E45"]
for pc, color in zip(v_plot["bodies"], colors):
    pc.set_facecolor(color)
v_plot["cmedians"].set_color(colors)
v_plot["cbars"].set_color(colors)
v_plot["cmins"].set_color(colors)
v_plot["cmaxes"].set_color(colors)


labels = ["EI", "PI", "UCB", "Simplex", "CCD"]
axes[2].legend(labels=labels, loc="lower left", prop={"size": 9})
axes[2].set_ylabel("Fraccion del maximo")
axes[2].set_xticklabels(["", "EI", "PI", "UCB", "Simplex", "CCD"])

# titulo
fig.suptitle(f"{n_dim} Factores", fontsize=14)
fig.tight_layout(pad=1.1)
# plt.show()

plt.savefig(f"opt_fig_ndim_{n_dim}", dpi=300)
