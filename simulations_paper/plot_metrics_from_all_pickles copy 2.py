import pickle
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os
import sys

FAMILY_COLORS = {
    'synthpop_norm': '#1f77b4',         # bleu foncé
    'synthpop_cart': '#a6c8ff',         # bleu ciel pastel
    'avatar': '#ff7f0e',                # orange
    'mixture': '#9467bd',               # violet
    'statis_dual_energy_weight': '#228B22',  # vert sapin
    'statis_dual_wass_weight':   '#228B22',  # vert sapin
    'statis_dual_energy_eigen': '#8FD694',   # vert pastel doux
    'statis_dual_wass_eigen':   '#8FD694',   # vert pastel doux
    'tabgan' : '#8B0000'
    }


DATASET_LABS = dict({"nls " : "Non Linear", 
                    "nls 200" : "Non Linear 200",
                    "nls 500" : "Non Linear 500",
                    "nls 5000" : "Non Linear 5000",
                    "lowcorr " : "Linear \n Low Correlation",
                    "highcorr " :"Linear \n Hight Correlation", 
                    "breast cancer" : "Breast Cancer",
                    "breast cancer " : "Breast Cancer"})

def get_family_color(method_name):
    for prefix, color in FAMILY_COLORS.items():
        if method_name.startswith(prefix):
            return color
    return '#BBBBBB'

def is_weightwass(method_name):
    return method_name.startswith("mixture_wasserstein") or method_name.startswith("statis_dual_wass")

def load_all_pickles(folder_path):
    return [
        (os.path.splitext(f)[0], pickle.load(open(os.path.join(folder_path, f), "rb")))
        for f in sorted(os.listdir(folder_path), key=lambda x: os.path.splitext(x)[0])
        if f.endswith(".pkl")
    ]

def load_all_pickles_order(folder_path, order_file):
    # Lire les noms des fichiers dans l'ordre spécifié
    with open(order_file, 'r') as f:
        ordered_names = [line.strip() for line in f if line.strip()]
    
    data_list = []
    for name in ordered_names:
        file_path = os.path.join(folder_path, f"{name}.pkl")
        if os.path.exists(file_path):
            with open(file_path, 'rb') as pf:
                data = pickle.load(pf)
                data_list.append((name, data))
        else:
            print(f"Warning: {file_path} not found.")
    
    return data_list



def summarize_metrics_grouped(data_list, methods, metrics, output_dir):
    """
    Crée un tableau récapitulatif des métriques (mean ± std) pour chaque dataset, méthode et métrique,
    et génère un fichier LaTeX avec coloration selon les règles.
    """
    rows = []
    for pkl_name, data in data_list:
        print(pkl_name)
        for metric_key in metrics:
            if metric_key not in data:
                continue
            for method in methods:
                if method not in data[metric_key]:
                    continue
                values = np.array(data[metric_key][method])
                mean = np.mean(values)
                std = np.std(values)
                rows.append({
                    "Dataset": DATASET_LABS[str.split(pkl_name, sep='best')[0].replace("_", " ")],
                    "Method": method,
                    "Metric": metric_key,
                    "Summary": f"{mean:.3f} ± {std:.3f}",
                    "Value": mean  # utile pour min/max
                })

    df = pd.DataFrame(rows)

    # Pivot : lignes = (Dataset, Method), colonnes = Metric
    pivot = df.pivot(index=["Dataset", "Method"], columns="Metric", values="Summary")

    # Fonction de coloration
    def colorize_group(values_str_list, metric):
        mean_vals = [float(v.split("±")[0].strip()) for v in values_str_list]
        max_val, min_val = max(mean_vals), min(mean_vals)
        colored = []
        for val_str, val in zip(values_str_list, mean_vals):
            if metric in ["Dcr", "NNDR"]:  # max = vert, min = rouge
                if val == max_val:
                    colored.append(f"\\cellcolor{{mygreen}}{val_str}")
                elif val == min_val:
                    colored.append(f"\\cellcolor{{myred}}{val_str}")
                else:
                    colored.append(val_str)
            else:  # Energy et Wasserstein : max = rouge, min = vert
                if val == max_val:
                    colored.append(f"\\cellcolor{{myred}}{val_str}")
                elif val == min_val:
                    colored.append(f"\\cellcolor{{mygreen}}{val_str}")
                else:
                    colored.append(val_str)
        return colored

    # Appliquer la coloration par dataset
    for dataset in pivot.index.get_level_values("Dataset").unique():
        subset_idx = pivot.loc[dataset].index  # méthodes de ce dataset
        for metric in pivot.columns:
            pivot.loc[(dataset, subset_idx), metric] = colorize_group(
                pivot.loc[(dataset, subset_idx), metric].tolist(),
                metric
            )

    # Sauvegarde LaTeX
    latex_path = os.path.join(output_dir, "summary_metrics.tex")
    with open(latex_path, "w") as f:
        f.write(pivot.to_latex(escape=False, multicolumn=True))

    return df, pivot

# Génère une couleur unique par méthode depuis tab10
def get_method_colors(methods):
    cmap = plt.get_cmap('tab10')
    return {method: cmap(i % 10) for i, method in enumerate(methods)}

def plot_metric_grouped_by_pickle(metric_keys, data_list, methods, method_labels, output_dir):
    import matplotlib.pyplot as plt
    import numpy as np
    import os

    num_pickles = len(data_list)
    num_methods = len(methods)

    fig, ax = plt.subplots(1, len(metric_keys), figsize=(45, 7 * num_pickles + 1))

    # Ajustement des tailles
    group_height = 1.2  # Hauteur totale allouée par groupe

    box_height = group_height / num_methods * 0.95  # Hauteur des boxplots
    #box_height = group_height / num_methods * 0.9  # Hauteur de chaque boxplot

    # Position centrale pour chaque pickle
    y_positions = np.arange(num_pickles) * 1.5  # Étire les groupes

    for i,metric_key in enumerate(metric_keys):
        for m_idx, method in enumerate(methods):
            for p_idx, (pkl_name, data) in enumerate(data_list):
           
                if method not in data[metric_key]:
                    continue

                values = data[metric_key][method]
                offset = -group_height / 2 + (m_idx + 0.5) * box_height
                pos = y_positions[p_idx] + offset

                box = ax[i].boxplot(
                values,
                positions=[pos],
                vert=False,
                widths=box_height * 0.9,
                patch_artist=True,
                showfliers=False,
                boxprops=dict(
                    facecolor=get_family_color(method),
                    edgecolor=get_family_color(method),
                    linewidth=1.2
                ),
                whiskerprops=dict(linewidth=1.2, color=get_family_color(method)),
                capprops=dict(linewidth=1.2, color=get_family_color(method)),
                medianprops=dict(color='black', linewidth=2)
                )

                if is_weightwass(method):
                    box['boxes'][0].set_hatch('////')
                    box['boxes'][0].set_facecolor('#FFFFFF')
                    box['boxes'][0].set_edgecolor(get_family_color(method))
                    box['boxes'][0].set_linewidth(1.5)

        ax[i].set_yticks(y_positions)


    #ax.set_yticklabels([str.split(name, sep='_')[0]+ " "+str.split(name, sep='_')[1] for name, _ in data_list], fontsize=30)
        ax[i].set_yticklabels([DATASET_LABS[str.split(name, sep='best')[0].replace("_", " ")] for name, _ in data_list], fontsize=30)
    #print([str.split(name, sep='_')[0]+ " "+str.split(name, sep='_')[1] for name, _ in data_list])
        ax[i].set_xlabel("Metric Value", fontsize=35)
        ax[i].set_title(metric_key.replace("_", " ").title(), fontsize=38, fontweight='bold')
        ax[i].grid(axis='x', linestyle=':', alpha=0.4)

    handles = []
    for method in methods:
        color = get_family_color(method)
        label = method_labels[methods.index(method)]
        if is_weightwass(method):
            patch = plt.Rectangle((0, 0), 1.5, 1.5, facecolor='white',
                                  edgecolor=color, hatch='////', linewidth=1.9, label=label)
        else:
            patch = plt.Rectangle((0, 0), 1.5, 1.5, facecolor=color,
                                  edgecolor=color, linewidth=1, label=label)
        handles.append(patch)

        
    plt.subplots_adjust(bottom=0.25)
   
    ax_legend = fig.add_axes([0, 0, 1, 1])  # occupe toute la figure
    ax_legend.axis('off')
    ax_legend.legend(handles,
                method_labels,  # centre dans la figure
                ncol=3,
                fontsize=32,
                title="SDG and Superlearner methods",
                title_fontsize=35,frameon=True,
                fancybox=True,
                borderpad=1.2,
                framealpha=0.9,
                loc='lower center',
                bbox_to_anchor=(0.5, 0),
                edgecolor='gray'
                )
    
    plt.tight_layout(rect=[0, 0.3, 1, 1])
    print(f"Saved: grouped_by_pickle_"+ "_".join(f"{key}" for key in metric_keys) +".pdf")
    
    fig.savefig(os.path.join(output_dir, f"grouped_by_pickle_"+ "_".join(f"{key}" for key in metric_keys) +".pdf"), dpi=300)
    plt.close()
    
    print(f"Saved: grouped_by_pickle_"+ "_".join(f"{key}" for key in metric_keys) +".pdf")




def plot_all_grouped(folder_path, output_dir, order_file=False):
    os.makedirs(output_dir, exist_ok=True)
    if order_file==False:
        data_list = load_all_pickles(folder_path)
    else :
        data_list = load_all_pickles_order(folder_path, order_file)

    if not data_list:
        print("No pickle files found.")
        return


    try:
        dfnjzivnjz
    #    METHODS = data_list[0][1]["METHODS"]
    #    print(METHODS)
    #    METHOD_LABELS = data_list[0][1]["METHODS_LABS"]
    
    except:
        METHODS = [
            'tabgan', 'synthpop_norm', 'synthpop_cart', 'avatar_ncp=4',
            "mixture_wasserstein", "mixture_energy",
            "statis_dual_energy_eigen", "statis_dual_wass_eigen",
            "statis_dual_energy_weight", "statis_dual_wass_weight", 
            ]
        METHOD_LABELS = [
            'TabGAN', 'synthpop\n norm', 'synthpop\ncart', 'avatar\n ncp=4',
            "mixture\n wasserstein", "mixture\n energy",
            "statis dual \n energy_eigen", "statis dual \n wass eigen",
            "statis dual \n energy weight", "statis dual \n wass weight"
            ]

        METHODS = [
            'tabgan', 'synthpop_norm', 'synthpop_cart', 'avatar_ncp=4',
            ]
        METHOD_LABELS = [
            'TabGAN', 'synthpop\n norm', 'synthpop\ncart', 'avatar\n ncp=4',
            ]
        
       

    metrics = ["metric_energy_scaled", "metric_wasserstein_scaled", "metric_dcr", "metric_nndr"]
    summarize_metrics_grouped(data_list, METHODS, metrics, output_dir)

    #for metric_key in metrics:
    plot_metric_grouped_by_pickle(["metric_energy_scaled", "metric_wasserstein_scaled"], data_list, METHODS, METHOD_LABELS, output_dir)
    plot_metric_grouped_by_pickle([ "metric_dcr", "metric_nndr"], data_list, METHODS, METHOD_LABELS, output_dir)

if __name__ == "__main__":
    if len(sys.argv) < 1:
        print("Usage: python plot_grouped_scores_from_folder.py path/to/folder output_dir")
    else:
        folder = sys.argv[1]
        output_dir = sys.argv[2]
        if len(sys.argv) ==3:
            plot_all_grouped(folder, output_dir)

        else :
            order_file = sys.argv[3]
            plot_all_grouped(folder, output_dir, order_file)
            

