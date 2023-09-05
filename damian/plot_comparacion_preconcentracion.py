import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


data = {
    "Elemento": [
        "Cd",
        "Mn",
        "Pb",
        "Sb",
        "Cd",
        "Mn",
        "Pb",
        "Sb",
    ],
    "Metodo de optizacion": [
        "CCD",
        "CCD",
        "CCD",
        "CCD",
        "BO",
        "BO",
        "BO",
        "BO",
    ],
    "Numero de experimentos": [19, 19, 19, 19, 10, 12, 10, 11],
}
df = pd.DataFrame(data)

sns.barplot(
    data=df, x="Elemento", y="Numero de experimentos", hue="Metodo de optizacion"
)


# ticks
yint = np.arange(0, 19, 2)
plt.title("Economia de los metodos de optimizacion.")
plt.legend(bbox_to_anchor=(1, -0.05), ncol=2, fancybox=True)
plt.yticks(yint)
plt.tight_layout()
plt.savefig(f"bayopt_bar", dpi=300)
plt.show()
