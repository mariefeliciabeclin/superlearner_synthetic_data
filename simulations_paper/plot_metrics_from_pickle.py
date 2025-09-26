# plot_scores_from_pickle.py
import pickle
import matplotlib.pyplot as plt
import numpy as np
import os
import sys


FAMILY_COLORS = {'synthpop_norm': '#1f77b4',         # bleu foncé
    'synthpop_cart': '#a6c8ff',         # bleu ciel pastel
    'avatar': '#ff7f0e',                # orange
    'mixture': '#9467bd',               # violet
    'statis_dual_energy_weight': '#228B22',  # vert sapin
    'statis_dual_wass_weight':   '#228B22',  # vert sapin
    'statis_dual_energy_eigen': '#8FD694',   # vert pastel doux
    'statis_dual_wass_eigen':   '#8FD694',   # vert pastel doux
    }



def get_family_color(method_name):
    for prefix, color in FAMILY_COLORS.items():
        if method_name.startswith(prefix):
            return color
    return '#BBBBBB'

def is_weightwass(method_name):
    return method_name.startswith("mixture_wasserstein") or method_name.startswith("statis_dual_wass")

def to_percentage(count_dict, total):
    return {k: (v / total) * 100 for k, v in count_dict.items()}

def plot_metric(ax, values, title, color, labels_methods):
    filtered = {k: v for k, v in values.items() if v > 0}
    sorted_items = sorted(filtered.items(), key=lambda x: x[1], reverse=True)
    methods, heights = zip(*sorted_items)
    labels = [labels_methods.get(m, m) for m in methods]

    x = np.arange(len(labels))
    bars = ax.bar(x, heights, color=color, alpha=0.85, edgecolor="black", width=0.6, linewidth=1)

    for bar in bars:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, h + 1,
                f"{h:.1f}%", ha='center', va='bottom',
                fontsize=15, fontweight='bold', color="black")

    ax.set_title(title, fontsize=18, fontweight="bold", pad=10, color="black")
    ax.set_ylim(0, 100)
    ax.set_yticks(np.arange(0, 110, 20))
    ax.set_ylabel("Percentage (%)", fontsize=15, color="black")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=40, ha="right", fontsize=16, color="black")
    ax.tick_params(axis='y', colors='black')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(True)
    ax.spines['bottom'].set_visible(True)
    ax.grid(axis='y', linestyle='--', alpha=0.4)

def pretty_boxplot(ax, data_dict, methods, labels, title):
    data = [data_dict[m] for m in methods]

    box = ax.boxplot(
        data,
        patch_artist=True,
        widths=0.4,
        showfliers=False,
        boxprops=dict(linewidth=1.0),
        whiskerprops=dict(linewidth=1.0),
        capprops=dict(linewidth=1.0),
        medianprops=dict(color='black', linewidth=2)
    )

    for patch, method in zip(box['boxes'], methods):
        patch.set_facecolor(get_family_color(method))
        patch.set_alpha(0.6)
        patch.set_edgecolor("black")
        patch.set_linewidth(1.0)
        if is_weightwass(method):
            patch.set_hatch('////')
            patch.set_edgecolor("black")

    ax.set_title(title, fontsize=18, fontweight="bold")
    ax.set_axisbelow(True)
    ax.grid(axis='y', linestyle=':', alpha=0.5)
    ax.set_facecolor('#f8f8f8')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(True)
    ax.spines['bottom'].set_visible(True)
    ax.tick_params(axis='x', labelsize=11)




def plot_all(pkl_path, data_type= 'lll', output_dir = "/home/mfbeclin/synthetic_data/synthetic_data/synthetic_proj_python/simulations_paper/res_image"):
    with open(pkl_path, "rb") as f:
        data = pickle.load(f)

    os.makedirs(output_dir, exist_ok=True)
    base_name = os.path.splitext(os.path.basename(pkl_path))[0]

    count_energy = data["count_energy"]
    count_wasserstein = data["count_wasserstein"]
    count_pmse = data["count_pmse"]
    count_nndr = data["count_nndr"]

    total = sum(count_energy.values())
    count_energy_pct = to_percentage(count_energy, total)
    count_wasserstein_pct = to_percentage(count_wasserstein, total)
    count_pmse_pct = to_percentage(count_pmse, total)
    count_nndr_pct = to_percentage(count_nndr, total)

    try :
        METHODS = data["METHODS"]
        METHODS_LABELS = data["METHODS_LABS"]
       
    except:
        METHODS = [
        'synthpop_norm', 'synthpop_cart', 'avatar_ncp=4',
        "mixture_wasserstein", "mixture_energy",
        "statis_dual_energy_eigen", "statis_dual_wass_eigen",
        "statis_dual_energy_weight", "statis_dual_wass_weight"
        ]

        METHODS_LABELS = [
            'synthpop\n norm', 'synthpop\ncart', 'avatar\n ncp=4',
            "mixture\n wasserstein", "mixture\n energy",
            "statis dual \n energy_eigen", "statis dual \n wass eigen",
            "statis dual \n energy weight", "statis dual \n wass weight"
        ]
    

    METHODS_LABELS_MAP = dict(zip(METHODS, METHODS_LABELS))
   

    # Bar plot (metric counts)
    fig, axs = plt.subplots(2, 2, figsize=(13, 10))
    metrics = [
        (count_energy_pct, "Energy Distance", axs[0, 0], "#6699cc"),
        (count_wasserstein_pct, "Wasserstein Distance", axs[0, 1], "#ff9966"),
        (count_nndr_pct, "NNDR", axs[1, 0], "#66cc99"),
        (count_pmse_pct, "PMSE (RF)", axs[1, 1], "#cc6666"),
    ]

    for values, title, ax, color in metrics:
        if values:
            plot_metric(ax, values, title, color, METHODS_LABELS_MAP)

    plt.suptitle("Privacy Metric Counts by Method", fontsize=15, fontweight="bold")
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(os.path.join(output_dir, base_name + "_counts_plot.pdf"), dpi=300)
    plt.close()

    # Boxplots
    fig, axes = plt.subplots(2, 2, figsize=(13, 13))

    pretty_boxplot(axes[0, 0], data["metric_energy_scaled"], METHODS, METHODS_LABELS, "Energy distance")
    pretty_boxplot(axes[1, 0], data["metric_wasserstein_scaled"], METHODS, METHODS_LABELS, "Wasserstein distance")
    pretty_boxplot(axes[0, 1], data["metric_dcr"], METHODS, METHODS_LABELS, r"$5^{th}$ percentile DCR")
    pretty_boxplot(axes[1, 1], data["metric_nndr"], METHODS, METHODS_LABELS, r"$5^{th}$ percentile NNDR")

    for ax in axes[0]:
        ax.set_xticklabels([])
    for ax in axes[1]:
        ax.set_xticklabels(METHODS_LABELS, rotation=45, ha='right', fontsize=12)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, base_name + "_scores_plot.pdf"), dpi=300)
    plt.close()
    print(f"Plots saved to {output_dir}")


    if data_type == "lele":

     # Boxplots
        fig, axes = plt.subplots(2, 2, figsize=(13, 13))

        pretty_boxplot(axes[0, 0], data["metric_energy_scaled"], METHODS, METHODS_LABELS, "Energy distance")
        pretty_boxplot(axes[1, 0], data["metric_wasserstein_scaled"], METHODS, METHODS_LABELS, "Wasserstein distance")
        pretty_boxplot(axes[0, 1], data["metric_dcr"], METHODS, METHODS_LABELS, r"$5^{th}$ percentile DCR")
        pretty_boxplot(axes[1, 1], data["metric_nndr"], METHODS, METHODS_LABELS, r"$5^{th}$ percentile NNDR")

        for ax in axes[0]:
            ax.set_xticklabels([])
        for ax in axes[1]:
            ax.set_xticklabels(METHODS_LABELS, rotation=45, ha='right', fontsize=12)

        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, base_name + "_scores_plot.pdf"), dpi=300)
        plt.close()
        print(f"Plots saved to {output_dir}")

# CLI usage
if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python plot_scores_from_pickle.py path/to/file.pkl output_dir")
    else:
        pkl_path = sys.argv[1]
        data_type = sys.argv[2]
        output_dir = sys.argv[3]
        plot_all(pkl_path= pkl_path, data_type = data_type, output_dir = output_dir )
