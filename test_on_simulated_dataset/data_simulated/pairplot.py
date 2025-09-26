import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import pearsonr

# Exemple de fonction pour annoter la corrélation
def corrfunc(x, y, color, label=None, ax=None):
    """Affiche le coefficient de corrélation de Pearson dans les sous-graphiques inférieurs."""
    r, _ = pearsonr(x, y)
    ax = ax or plt.gca()
    ax.annotate(f"ρ = {r:.2f}", xy=(0.1, 0.9), xycoords=ax.transAxes, fontsize=20, color="red")

# -----------------------------
# Exemple avec ton DataFrame
# -----------------------------
# df = pd.read_csv("mon_fichier.csv")  # ton DataFrame
# Ici je fais un exemple aléatoire :

df = pd.read_csv("/home/marie-felicia/synthetic_data/synthetic_proj_python/test_on_simulated_dataset/data_simulated/simulated_data_lowcorr.csv", index_col=0)

# Pairplot
g = sns.pairplot(df, corner=True, diag_kind="kde",
                 plot_kws=dict(marker=".", linewidth=1, alpha=0.5))

# Ajout des corrélations dans la matrice inférieure
g.map_lower(corrfunc)

g.savefig("/home/marie-felicia/synthetic_data/synthetic_proj_python/test_on_simulated_dataset/data_simulated/pairplot_lowcorr.pdf")


plt.show()
