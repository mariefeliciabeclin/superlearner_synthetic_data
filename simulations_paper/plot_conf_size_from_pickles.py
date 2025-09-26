import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pickle
from pathlib import Path
import statsmodels.api as sm
from matplotlib.patches import Patch

FAMILY_COLORS = {
    'synthpop_norm': '#1f77b4',
    'synthpop_cart': '#a6c8ff',
    'avatar': '#ff7f0e',
    'mixture': '#9467bd',
    'statis_dual_energy_weight': '#228B22',
    'statis_dual_wass_weight': '#228B22',
    'statis_dual_energy_eigen': '#8FD694',
    'statis_dual_wass_eigen': '#8FD694',
}

def get_family_color(method_name):
    for prefix, color in FAMILY_COLORS.items():
        if method_name.startswith(prefix):
            return color
    return '#BBBBBB'

def is_weightwass(method_name):
    return method_name.startswith("mixture_wasserstein") or method_name.startswith("statis_dual_wass")

BASE_DIR = Path("/home/mfbeclin/synthetic_data/synthetic_data/synthetic_proj_python")
OUT_PATH = BASE_DIR / "simulations_paper/res_metrics/with_synpop_cart"

OUT_PATH = BASE_DIR / "simulations_paper/res_metrics"


OUT_PATH_IMAGE = BASE_DIR / "simulations_paper/res_image/best_with_synpopcart"

df_true_high = pd.read_csv(BASE_DIR / "test_on_simulated_dataset/data_simulated/simulated_data_highcorr.csv", index_col=0)
df_true_low = pd.read_csv(BASE_DIR / "test_on_simulated_dataset/data_simulated/simulated_data_lowcorr.csv", index_col=0)

VAR_LIST = ['X0', 'X1', 'X2', 'X3']

conf_size_high = sm.OLS(df_true_high['Y'], df_true_high[VAR_LIST]).fit().conf_int(alpha=0.05).apply(lambda x: abs(x[1] - x[0]), axis=1).tolist()
conf_size_low = sm.OLS(df_true_low['Y'], df_true_low[VAR_LIST]).fit().conf_int(alpha=0.05).apply(lambda x: abs(x[1] - x[0]), axis=1).tolist()

def load_biases(pkl_path):
    with open(pkl_path, "rb") as f:
        data = pickle.load(f)
        print(data)
    return data["res_conf_size"]

def extract_bias_stats(res_conf_size, var_list):
    stats = {var: {} for var in var_list}
    for method, df_list in res_conf_size.items():
        df = pd.DataFrame([df for df in df_list], columns= var_list)
        for var in var_list:
            stats[var][method] = df[var]
    return stats

def plot_bias_two_rows(lowcorr_stats, highcorr_stats, methods, var_list, output_path=None):
    n_vars = len(var_list)
    n_methods = len(methods)

    fig, axs = plt.subplots(2, n_vars, figsize=(7 * n_vars, 1 * n_methods + 5), sharey=True)

    if n_vars == 1:
        axs = np.array([[axs[0]], [axs[1]]])

    spacing = 0.15
    box_height = 0.1

    for j, var in enumerate(var_list):
        for row, (stats, conf_size, row_label) in enumerate([
            (lowcorr_stats, conf_size_low, "Lowcorr"),
            (highcorr_stats, conf_size_high, "Highcorr")
        ]):
            ax = axs[row, j]
            yticks = []
            yticklabels = []

            for i, method in enumerate(methods):
                pos = i * spacing
                if method in stats[var]:
                    vals = stats[var][method]
                    box = ax.boxplot(
                        vals,
                        positions=[pos],
                        vert=False,
                        widths=box_height,
                        patch_artist=True,
                        showfliers=False,
                        boxprops=dict(facecolor=get_family_color(method), edgecolor=get_family_color(method), linewidth=1.2),
                        whiskerprops=dict(linewidth=1.2, color=get_family_color(method)),
                        capprops=dict(linewidth=1.2, color=get_family_color(method)),
                        medianprops=dict(color='black', linewidth=2)
                    )
                    if is_weightwass(method):
                        box['boxes'][0].set_hatch('////')
                        box['boxes'][0].set_facecolor('#FFFFFF')
                        box['boxes'][0].set_edgecolor(get_family_color(method))
                        box['boxes'][0].set_linewidth(1.5)

                    yticks.append(pos)
                    yticklabels.append("")

            # Lignes verticales (intervalle de confiance)
            
            ax.axvline(x=conf_size[j] , color='grey', linestyle='--')
        

            ax.set_yticks(yticks)
            ax.set_yticklabels(yticklabels)
            ax.set_facecolor("#f8f8f8")
            ax.grid(axis='x', linestyle=':', alpha=0.4)
            

            # Ajout du label de ligne (Lowcorr/Highcorr)
            if j==0:
                ax.text(
                    -0.05, 0.5, row_label,
                    transform=ax.transAxes,
                    fontsize=15,
                    fontweight='bold',
                    va='center',
                    ha='right',
                    rotation=90
                    )

    # Récupérer la position du x=0 (en pixels) pour chaque axe
   
        axs[0, j].set_title(
    fr"$\hat{{\beta}}_{{{j}}}^{{\mathrm{{synth}}}} - \hat{{\beta}}_{{{j}}}^{{\mathrm{{original}}}}$",
    fontsize=16,
    fontweight='bold'
)

    # Légende
    legend_elements = [
        Patch(facecolor=color, edgecolor='black', label=name)
        for name, color in FAMILY_COLORS.items()
    ]
    
    handles = []
    for method in methods:
        color = get_family_color(method)
        label = METHODS_LABELS[methods.index(method)]
        if is_weightwass(method):
            patch = plt.Rectangle((0, 0), 1.5, 1.5, facecolor='white',
                                  edgecolor=color, hatch='////', linewidth=1.9, label=label)
        else:
            patch = plt.Rectangle((0, 0), 1.5, 1.5, facecolor=color,
                                  edgecolor=color, linewidth=1, label=label)
        handles.append(patch)

    fig.legend(
        handles,
        METHODS_LABELS,
        title="SDG and Superlearner methods",
        loc='upper center',
        bbox_to_anchor=(0.5, -0.05),  # Ajuste si besoin
        ncol=5,  # Divise en 2 lignes si tu as ~10 méthodes
        fontsize=14,
        title_fontsize=15,
        frameon=True,
        fancybox=True,
        borderpad=1.2,
        framealpha=0.9,
        edgecolor='gray'
        )

 
    plt.tight_layout(rect=[0, 0.01, 1, 0.99])
    if output_path:
        plt.savefig(output_path / "conf_size_rows.pdf", bbox_inches="tight")
        print(f"Plot saved to: {output_path / 'bias_two_rows.pdf'}")
    else:
        plt.show()
    plt.close()

if __name__ == "__main__":
    res_high_conf_size = load_biases(OUT_PATH / "highcorr_best_n=10.pkl")
    res_low_conf_size = load_biases(OUT_PATH / "lowcorr_best_n=10.pkl")

    try:
        METHODS = res_high["METHODS"]
        METHODS_LABELS = res_high["METHODS_LABELS"]

    except:
        METHODS = [
            'synthpop_norm',  'synthpop_cart', 'tabgan', 'avatar_ncp=4',
            "mixture_wasserstein", "mixture_energy",
            "statis_dual_energy_eigen", "statis_dual_wass_eigen",
            "statis_dual_energy_weight", "statis_dual_wass_weight"
        ]
        METHODS_LABELS = ['synthpop\n norm', 'synthpop \n cart','tabgan', 'avatar\n ncp=4',
            "mixture \n wasserstein", "mixture\n energy",
            "statis_dual \n energy_eigen", "statis_dual \n wass_eigen",
            "statis_dual\n energy_weight", "statis_dual \n wass_weight"
        ]

    low_stats = extract_bias_stats(res_high_conf_size, VAR_LIST)
    high_stats = extract_bias_stats(res_low_conf_size, VAR_LIST)

    plot_bias_two_rows(
        lowcorr_stats=low_stats,
        highcorr_stats=high_stats,
        methods=METHODS,
        var_list=VAR_LIST,
        output_path=OUT_PATH_IMAGE
    )
