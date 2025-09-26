import pickle
import matplotlib.pyplot as plt
import numpy as np
import os

# Chargement des données
with open("/home/mfbeclin/synthetic_data/synthetic_data/synthetic_proj_python/simulations_paper/res_metrics/res_aal_metrics_n=3.pkl", "rb") as f:
    all_dicts = pickle.load(f)

# Dictionnaires des métriques
metric_energy_scaled = {
    "lowcorr": all_dicts["metric_energy_scaled_lowcorr"],
    "nls": all_dicts["metric_energy_scaled_nls"],
    "breast_cancer": all_dicts["metric_energy_scaled_bc"]
}
metric_wass_scaled = {
    "lowcorr": all_dicts["metric_wasserstein_scaled_lowcorr"],
    "nls": all_dicts["metric_wasserstein_scaled_nls"],
    "breast_cancer": all_dicts["metric_wasserstein_scaled_bc"]
}

FAMILY_COLORS = {
    'synthpop_norm':      '#1f77b4',  # Bleu foncé
    'synthpop_cart':      '#a6c8ff',  # Bleu clair

    'avatar_ncp=2':   '#ff6f00',  # Orange doré
    
    'avatar_ncp=4':  '#ff851b',  # Orange intense
    'avatar_ncp=5':   '#ffae42',  # Orange doré
 

    'synthcity_ctgan':    '#ff9999',  # Rouge clair
    'sdv_ctgan':          '#b30000',  # Rouge foncé
}

FAMILY_HATCHES = {
    'avatar_ncp=2_k=10':  '//',
    'avatar_ncp=4_k=10':  '//',
    'avatar_ncp=5_k=10':  '//',
    # Les autres peuvent rester sans hachures
}



def get_family_color(method_name):
    for prefix, color in FAMILY_COLORS.items():
        if method_name.startswith(prefix):
            return color
    return '#BBBBBB'

def is_weightwass(method_name):
    return 'weight' in method_name.lower()

def plot(dict_value, title, output_dir, methods=None):
    data_list = list(dict_value.items())
    num_pickles = len(data_list)

    # Utiliser les méthodes spécifiées ou toutes
    all_methods = list(next(iter(dict_value.values())).keys())
    selected_methods = methods if methods is not None else all_methods
    num_methods = len(selected_methods)

    fig, ax = plt.subplots(figsize=(20, 16))
    group_height = 1.2
    box_height = group_height / num_methods * 0.95
    y_positions = np.arange(num_pickles) * 1.5

    for p_idx, (dataset_name, methods_dict) in enumerate(data_list):
        for m_idx, method in enumerate(selected_methods):
            if method not in methods_dict:
                continue  # Skip if method not in this dataset

            values = methods_dict[method]
            offset = -group_height / 2 + (m_idx + 0.5) * box_height
            pos = y_positions[p_idx] + offset
            hatch = FAMILY_HATCHES.get(method, '') 
    
            box = ax.boxplot(
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
                medianprops=dict(color='black', linewidth=2), 
                
            )

            if hatch == '//':
                
                box['boxes'][0].set_hatch('//')
                box['boxes'][0].set_facecolor('#FFFFFF')
                box['boxes'][0].set_edgecolor(get_family_color(method))
                box['boxes'][0].set_linewidth(1.5)
                print('yes')

                
            if is_weightwass(method):
                box['boxes'][0].set_hatch('////')
                box['boxes'][0].set_facecolor('#FFFFFF')
                box['boxes'][0].set_edgecolor(get_family_color(method))
                box['boxes'][0].set_linewidth(1.5)

    ax.set_yticks(y_positions)
    ax.set_yticklabels([name for name, _ in data_list], fontsize=35)
    ax.set_xlabel("Metric Value", fontsize=20)
    ax.set_title(title, fontsize=30, fontweight='bold')
    ax.grid(axis='x', linestyle=':', alpha=0.4)

    # Légende
    handles = []
    for method in selected_methods:
        color = get_family_color(method)
        if FAMILY_HATCHES.get(method, '') =="//":
            patch = plt.Rectangle((0, 0), 1.5, 1.5, facecolor='white',
                                  edgecolor=color, hatch='//', linewidth=1.9, label=method)
        else:
            patch = plt.Rectangle((0, 0), 1.5, 1.5, facecolor=color,
                                  edgecolor=color, linewidth=1, label=method)
        handles.append(patch)

    ax.legend(
        handles,
        selected_methods,
        title="SDG and Superlearner methods",
        loc='upper center',
        bbox_to_anchor=(0.5, -0.05),
        ncol=5,
        fontsize=20,
        title_fontsize=20,
        frameon=True,
        fancybox=True,
        borderpad=1.2,
        framealpha=0.9,
        edgecolor='gray'
    )

    plt.tight_layout()
 
    os.makedirs(output_dir, exist_ok=True)
    filename = f"{'best' if methods else 'all_methods'}_{title.replace(' ', '_')}.pdf"
    fig.savefig(os.path.join(output_dir, filename), dpi=300)
    plt.close()

# Exemple d'appel
METHODS_best = [ 'synthpop_norm', 'synthpop_cart', 'avatar_ncp=4_k=10','avatar_ncp=4_k=5','avatar_ncp=5_k=10','avatar_ncp=5_k=5' ]

plot(metric_energy_scaled, "Energy Distance (scaled)", output_dir="/home/mfbeclin/synthetic_data/synthetic_data/synthetic_proj_python/simulations_paper/res_image")
plot(metric_wass_scaled, "Wasserstein Distance (scaled)", output_dir="/home/mfbeclin/synthetic_data/synthetic_data/synthetic_proj_python/simulations_paper/res_image")
plot(metric_energy_scaled, "Energy Distance (scaled)", output_dir="/home/mfbeclin/synthetic_data/synthetic_data/synthetic_proj_python/simulations_paper/res_image", methods =METHODS_best )
plot(metric_wass_scaled, "Wasserstein Distance (scaled)", output_dir="/home/mfbeclin/synthetic_data/synthetic_data/synthetic_proj_python/simulations_paper/res_image", methods = METHODS_best)


