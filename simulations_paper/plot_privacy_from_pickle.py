import pickle
import matplotlib.pyplot as plt
import numpy as np
import sys
import os

from matplotlib.colors import to_rgba


def is_weightwass(method_name):
    return method_name.startswith("mixture_wasserstein") or method_name.startswith("statis_dual_wass")

def plot_from_pickle(pkl_path, output_path=None, dataset_label="Dataset"):
    with open(pkl_path, "rb") as f:
        all_dicts = pickle.load(f)

    METHODS = [
        'synthpop_norm', 'synthpop_cart', 'avatar_ncp=4',
        "tabgan",
        "mixture_wasserstein", "mixture_energy",
        "statis_dual_energy_eigen", "statis_dual_wass_eigen",
        "statis_dual_energy_weight", "statis_dual_wass_weight"
    ]
    METHODS_LABELS= [
        'synthpop\n norm', 'synthpop\ncart', 'avatar\n ncp=4',
        "tabgan",
        "mixture\n wasserstein", "mixture\n energy",
        "statis dual \n energy_eigen", "statis dual \n wass eigen",
        "statis dual \n energy weight", "statis dual \n wass weight"
    ]

    # Définir les couleurs par famille (plusieurs nuances si besoin)
    FAMILY_COLORS = {
        'synthpop_norm': '#1f77b4',         # bleu foncé
        'synthpop_cart': '#a6c8ff',         # bleu ciel pastel
        'avatar': '#ff7f0e',                # orange
        'mixture': '#9467bd',               # violet
        'statis_dual_energy_weight': '#228B22',  # vert sapin
        'statis_dual_wass_weight':   '#228B22',  # vert sapin
        'statis_dual_energy_eigen': '#8FD694',   # vert pastel doux
        'statis_dual_wass_eigen':   '#8FD694',   # vert pastel doux
        }


    # Associe chaque méthode à sa couleur
    method_colors = {}
    family_counts = {k: 0 for k in FAMILY_COLORS.keys()}
    for method in METHODS:
        for family in FAMILY_COLORS:
            if method.startswith(family):
                idx = family_counts[family]
                color_list = FAMILY_COLORS[family]
                color = FAMILY_COLORS[family]
                method_colors[method] = color
                #method_colors[method] = color_list[idx % len(color_list)]
                family_counts[family] += 1
                break
        else:
            method_colors[method] = '#999999'  # fallback

    percentiles = np.array([0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95]) * 100
    fig, ax = plt.subplots(1, 2, figsize=(20, 10))

    num_methods = len(METHODS)
    group_width = 4.0
    offset = group_width / num_methods

    for i, metric_name in enumerate(["metric_dcr_all", "metric_nndr_all"]):
        metric_all = all_dicts[metric_name]

        for j, method in enumerate(METHODS):
            if method not in metric_all:
                continue

            data = metric_all[method]
            data_by_percentile = [list(p) for p in zip(*data)]
            positions = percentiles + (j - num_methods / 2) * offset + offset / 2

            color = method_colors[method]

            ax[i].boxplot(
                data_by_percentile,
                positions=positions,
                widths=offset * 0.9,
                patch_artist=True,
                boxprops=dict(facecolor=to_rgba(color, alpha=0.45), edgecolor=color),
                medianprops=dict(color='black', linewidth=2),
                whiskerprops=dict(color=color),
                capprops=dict(color=color),
                flierprops=dict(marker='o', markersize=3, linestyle='none', color=color, alpha=0.6)
            )

            medians = [np.median(p) for p in data_by_percentile]
            if is_weightwass(method) : 
                ax[i].plot(positions, medians, color=color, linewidth=3, label=METHODS_LABELS[j], linestyle='--')
                #ax[i].boxplot['boxes'][0].set_hatch('////')
            else :
                ax[i].plot(positions, medians, color=color, linewidth=3, label=METHODS_LABELS[j])
                
        ax[i].set_title("DCR" if i == 0 else "NNDR", fontsize=20)
        ax[i].set_xticks(percentiles)
        ax[i].set_xticklabels([f"{int(p)}%" for p in percentiles], fontsize=20)
        ax[i].set_xlabel("Percentile", fontsize=18)
        ax[i].set_ylabel(dataset_label, fontsize=18)
        ax[i].grid(axis='y', linestyle='--', alpha=0.4)

        ax[i].spines['top'].set_visible(False)
        ax[i].spines['right'].set_visible(False)
        ax[i].spines['left'].set_visible(True)
        ax[i].spines['bottom'].set_visible(True)

    # Légende à droite
    handles, labels = ax[0].get_legend_handles_labels()
    fig.legend(
        handles, labels,
        loc='center left',
        bbox_to_anchor=(1.02, 0.5),
        borderaxespad=0.0,
        fontsize=13,
        title="SDG Methods",
        title_fontsize=15,
        frameon=True
    )

    plt.subplots_adjust(right=0.8)

    print(os.path.splitext(pkl_path)[0].split("/")[-1])
    output_path = output_path+"/"+os.path.splitext(pkl_path)[0].split("/")[-1] + "_privacy_plot.pdf"

    plt.tight_layout() 
    plt.savefig(output_path, bbox_inches='tight')
    
    print(f"Saved: {output_path}")
    plt.close()

# Usage
if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python plot_privacy_from_pickle.py path/to/file.pkl [dataset_label]")
    else:
        pkl_file = sys.argv[1]
        output_path = sys.argv[2]
        dataset_name = sys.argv[3] if len(sys.argv) > 3 else ""
        plot_from_pickle(pkl_path=pkl_file, output_path=output_path, dataset_label=dataset_name)


        