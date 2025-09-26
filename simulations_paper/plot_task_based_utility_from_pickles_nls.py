import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pickle
from pathlib import Path
import statsmodels.api as sm
import sys

sys.path.append(str(Path(__file__).resolve().parent.parent))
from nonlinear_regression.nls import regression_nls


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
OUT_PATH = BASE_DIR / "simulations_paper/res_article/nls_version"
OUT_PATH_IMAGE = BASE_DIR / "simulations_paper/res_article/nls_version/images"

df_true_200 = pd.read_csv(BASE_DIR / "test_on_simulated_dataset/data_simulated/simulated_data_nls_200.csv")
df_true_500 = pd.read_csv(BASE_DIR / "test_on_simulated_dataset/data_simulated/simulated_data_nls_500.csv")
df_true_5000 = pd.read_csv(BASE_DIR / "test_on_simulated_dataset/data_simulated/simulated_data_nls_5000.csv")

params_orig_200 = regression_nls(df=df_true_200)
params_orig_500 = regression_nls(df=df_true_500)
params_orig_5000 = regression_nls(df=df_true_5000)

params_true = [2.5, 1.3, 2, 1.2, 0.03]
VAR_LIST = ['a', 'beta_0', 'beta_1', 'beta_2', 'beta_3']

def load_biases(pkl_path):
    with open(pkl_path, "rb") as f:
        data = pickle.load(f)
    return data["res"]

def extract_bias_stats(res, var_list):
    stats = {var: {} for var in var_list}
    for method, df_list in res.items():
        df = pd.concat([df for df in df_list])
        for var in var_list:
            stats[var][method] = df[var]
    return stats

import pickle
import pandas as pd
from pathlib import Path

OUT_PATH = Path("/home/mfbeclin/synthetic_data/synthetic_data/synthetic_proj_python/simulations_paper/res_article/nls_version")
VAR_LIST = ['a', 'beta_0', 'beta_1', 'beta_2', 'beta_3']

def load_biases(pkl_path):
    with open(pkl_path, "rb") as f:
        data = pickle.load(f)
    return data["res"]

def extract_bias_stats(res, var_list):
    stats = {var: {} for var in var_list}
    for method, df_list in res.items():
        df = pd.concat([df for df in df_list])
        for var in var_list:
            stats[var][method] = df[var]
    return stats

def summarize_stats(stats):
    summary_rows = []
    for var, methods_dict in stats.items():
        for method, values in methods_dict.items():
            mean = values.mean()
            std = values.std()
            median = values.median()
            q1 = values.quantile(0.25)
            q3 = values.quantile(0.75)
            n_missing = values.isna().sum()
            summary_rows.append({
                "Variable": var,
                "Method": method,
                "Mean": mean,
                "Std": std,
                "Median": median,
                "Q1": q1,
                "Q3": q3,
                "NaNs": n_missing
            })
    return pd.DataFrame(summary_rows)


def table_results():
# Charger les fichiers .pkl
    res_200 = load_biases(OUT_PATH / "nls_200_n=100.pkl")
    res_500 = load_biases(OUT_PATH / "nls_500_n=100.pkl")
    res_5000 = load_biases(OUT_PATH / "nls_5000_n=100.pkl")

# Extraire les statistiques
    nls_200_stats = extract_bias_stats(res_200, VAR_LIST)
    nls_500_stats = extract_bias_stats(res_500, VAR_LIST)
    nls_5000_stats = extract_bias_stats(res_5000, VAR_LIST)

# Créer les tableaux résumés
    summary_200 = summarize_stats(nls_200_stats)
    summary_500 = summarize_stats(nls_500_stats)
    summary_5000 = summarize_stats(nls_5000_stats)

# Ajouter une colonne "Sample Size"
    summary_200["N"] = 200
    summary_500["N"] = 500
    summary_5000["N"] = 5000

# Fusionner tous les tableaux
    summary_all = pd.concat([summary_200, summary_500, summary_5000], ignore_index=True)
    # Créer une colonne "M ± SD" et "Median [Q1, Q3]"
    summary_all["Mean±SD"] = summary_all.apply(lambda row: f"{row['Mean']:.3f} ± {row['Std']:.3f}", axis=1)
    summary_all["Median [Q1, Q3]"] = summary_all.apply(lambda row: f"{row['Median']:.3f} [{row['Q1']:.3f}, {row['Q3']:.3f}]", axis=1)

    # Sélectionner seulement les colonnes utiles pour le tableau
    table_df = summary_all.pivot(index=['Method', 'N'], columns='Variable', values='Mean±SD')

    # Générer LaTeX
    latex_table = table_df.to_latex(multicolumn=True)  # index=True pour garder Method et N
    with open(OUT_PATH / "regression_table_results.tex", "w") as f:
        f.write(latex_table)

    







def plot_bias_three_rows(nls_200_stats, nls_500_stats, nls_5000_stats, methods, methods_labels, var_list, output_path=None):
    n_vars = len(var_list)
    n_methods = len(methods)

    fig, axs = plt.subplots(3, n_vars, figsize=(7 * n_vars, 1.8 * n_methods + 5), sharey=True)

    if n_vars == 1:
        axs = np.array([[axs[0]], [axs[1]], [axs[2]]])

    spacing = 0.15
    box_height = 0.1

    # Calcul des limites en x par variable sans outliers
    xlims = {}
    for var in var_list:
        all_vals = []
        for stats in [nls_200_stats, nls_500_stats, nls_5000_stats]:
            for method in methods:
                if method in stats[var]:
                    vals = stats[var][method].dropna()
                    if not vals.empty:
                        all_vals.extend(vals.tolist())
        if len(all_vals) > 0:
            low = np.percentile(all_vals, 1)
            high = np.percentile(all_vals, 99)
            margin = 0.1 * (high - low) if high > low else 1
            xlims[var] = (low - margin, high + margin)
        else:
            xlims[var] = (-1, 1)

    for j, var in enumerate(var_list):
        for row, (stats, row_label, params_orig) in enumerate([
            (nls_200_stats, "nls_200", params_orig_200),
            (nls_500_stats, "nls_500", params_orig_500),
            (nls_5000_stats, "nls_5000", params_orig_5000),
        ]):
            ax = axs[row, j]
            yticks = []
            yticklabels = []

            # Lignes verticales sans légende
            ax.axvline(params_true[j], color='blue', linestyle='-', linewidth=2)
            val_orig = params_orig[var_list[j]][0]
            ax.axvline(val_orig, color='red', linestyle='--', linewidth=2)
            print( params_orig[var_list[3]][0])

            for i, method in enumerate(methods):
                pos = i * spacing
                if method in stats[var]:
                    vals = stats[var][method]
                    n_nans = vals.isna().sum()
                    vals_clean = vals.dropna()

                    box = ax.boxplot(
                        vals_clean,
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

                    if n_nans > 0:
                        median_x = np.median(vals_clean) if len(vals_clean) > 0 else 0
                        ax.text(
                            x=median_x + 0.02,
                            y=pos,
                            s=f"{n_nans} NaN",
                            va='center',
                            ha='left',
                            fontsize=30,
                            color='red'
                        )

                    yticks.append(pos)
                    yticklabels.append("")

            ax.set_yticks(yticks)
            ax.tick_params(axis='x', labelsize=34) 
            ax.set_yticklabels(yticklabels)
            ax.set_facecolor("#f8f8f8")
            ax.grid(axis='x', linestyle=':', alpha=0.4)
            ax.set_xlim(xlims[var])

            if j == 0:
                ax.text(
                    -0.05, 0.5, row_label,
                    transform=ax.transAxes,
                    fontsize=50,
                    fontweight='bold',
                    va='center',
                    ha='right',
                    rotation=90
                )

            if j==0 :
                axs[0, j].set_title(
                fr"$\hat{{a}}^{{\mathrm{{synth}}}}$",
                fontsize=50,
                fontweight='bold'
            )
            else : 
                axs[0, j].set_title(
                fr"$\hat{{\beta}}_{{{j-1}}}^{{\mathrm{{synth}}}}$",
                fontsize=50,
                fontweight='bold'
                )

    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch

# Légende lignes verticales
    legend_lines = [
    Line2D([0], [0], color='blue', lw=2, label='params_true'),
    Line2D([0], [0], color='red', lw=2, linestyle='--', label='params_orig'),
        ]

# Légende méthodes
    legend_methods = []
    for method, label in zip(methods, METHODS_LABELS):
        color = get_family_color(method)
        patch = Patch(facecolor=color, edgecolor=color, label=label)
        if is_weightwass(method):
            patch.set_facecolor('#FFFFFF')
            patch.set_hatch('////')
        legend_methods.append(patch)

    fig.legend( 
        handles=legend_lines + legend_methods,
        loc='upper center',
        bbox_to_anchor=(0.5, -0.05),
        ncol=5,
        fontsize=30,
        title="Legend",
        title_fontsize=34,
        frameon=True,
        fancybox=True,
        borderpad=1.2,
        framealpha=0.9,
        edgecolor='gray'
    )

    plt.tight_layout(rect=[0, 0.01, 1, 0.99])
    if output_path:
        plt.savefig(output_path / "bias_linear_regression.pdf", bbox_inches="tight")
        print(f"Plot saved to: {output_path / 'bias_linear_regression.pdf'}")
    else:
        plt.show()
    plt.close()


if __name__ == "__main__":
    res_200 = load_biases(OUT_PATH / "nls_200_n=100.pkl")
    res_500 = load_biases(OUT_PATH / "nls_500_n=100.pkl")
    res_5000 = load_biases(OUT_PATH / "nls_5000_n=100.pkl")

    try:
        METHODS = res_high["METHODS"]
        METHODS_LABELS = res_high["METHODS_LABELS"]
    except:
        METHODS = [
            'synthpop_norm', 'synthpop_cart', 'avatar_ncp=4',
            "mixture_wasserstein", "mixture_energy",
            "statis_dual_energy_eigen", "statis_dual_wass_eigen",
            "statis_dual_energy_weight", "statis_dual_wass_weight"
        ]
        METHODS_LABELS = [
            'synthpop\n norm', 'synthpop \n cart', 'avatar\n ncp=4',
            "mixture \n wasserstein", "mixture\n energy",
            "statis_dual \n energy_eigen", "statis_dual \n wass_eigen",
            "statis_dual\n energy_weight", "statis_dual \n wass_weight"
        ]

    nls_200_stats = extract_bias_stats(res_200, VAR_LIST)
    nls_500_stats = extract_bias_stats(res_500, VAR_LIST)
    nls_5000_stats = extract_bias_stats(res_5000, VAR_LIST)

    plot_bias_three_rows(
        nls_200_stats=nls_200_stats,
        nls_500_stats=nls_500_stats,
        nls_5000_stats=nls_5000_stats,
        methods=METHODS,
        methods_labels=METHODS_LABELS,
        var_list=VAR_LIST,
        output_path=OUT_PATH_IMAGE
    )
    table_results()