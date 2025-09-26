import pickle
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import numpy as np
import sys
import os

from matplotlib.colors import to_rgba


def is_weightwass(method_name):
    return method_name.startswith("mixture_wasserstein") or method_name.startswith("statis_dual_wass")

def plot_from_pickle(pkl_path, output_path=None, dataset_name="lowcorr"):
    with open(pkl_path, "rb") as f:
        data = pickle.load(f)
        

    with open(pkl_path_original, "rb") as f:
        data_original = pickle.load(f)


    try :
        METHODS = data["METHODS"]
        METHODS_LABELS = data["METHODS_LABS"]
    except:
        METHODS = [
        'synthpop_norm', 'synthpop_cart', 'avatar_ncp=4',
        "mixture_energy", "mixture_wasserstein",
        "statis_dual_energy_eigen", "statis_dual_wass_eigen",
        "statis_dual_energy_weight", "statis_dual_wass_weight"
        ]
        METHODS_LABELS= [
        'synthpop\n norm', 'synthpop\ncart', 'avatar\n ncp=4',
        "mixture\n energy", "mixture\n wasserstein",
        "statis dual \n energy_eigen", "statis dual \n wass eigen",
        "statis dual \n energy weight", "statis dual \n wass weight"
        ]

    # Définir les couleurs par famille (plusieurs nuances si besoin)
    FAMILY_COLORS = {
        'synthpop_norm': '#1f77b4',              # bleu foncé
        'synthpop_cart': '#4f9de2',              # bleu moyen (tonalité différente)
        'avatar': '#ff7f0e',                     # orange
        'mixture': '#9467bd',                    # violet

    # regrouper les "statis_dual_energy" et "statis_dual_wass"
        'statis_dual_energy_weight': '#228B22',  # vert sapin foncé
        'statis_dual_wass_weight':   '#2e8b57',  # vert forêt

        'statis_dual_energy_eigen': '#8FD694',   # vert pastel clair
        'statis_dual_wass_eigen':   '#b3e6b3',   # vert très clair
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

    data_privacy = dict({})
    data_metric_energy = dict({})
    data_metric_wass = dict({})
    data_privacy_rel = dict({})
    data_metric_wass_rel = dict({})
    data_metric_energy_rel = dict({})
    data_to_plot = {
        'energy_distance': [],
        'wasserstein_distance': [],
        'privacy': [],
        'Category': [],
    }


    data_to_plot_rel = {
        'wasserstein_distance': [],
        'energy_distance': [],
        'privacy': [],
        'Category': [],
    }



    for method in METHODS:
        
        #data_privacy[method] = [d/data_privacy_original for d in data["metric_nndr"][method]]
        #data_metric[method] = [d/data_metric_original-1 for d in data["metric_wasserstein_scaled"][method]]


        data_privacy_original = np.mean(data_original["metric_nndr"])
        data_wass_original = np.mean(data_original["metric_wasserstein_scaled"])
        data_energy_original = np.mean(data_original["metric_energy_scaled"])



        data_privacy[method] = [d for d in data["metric_nndr"][method]]
        data_metric_energy[method] = [d for d in data["metric_energy_scaled"][method]]
        data_metric_wass[method] = [d for d in data["metric_wasserstein_scaled"][method]]

        data_privacy_rel[method] = [d/data_privacy_original for d in data["metric_nndr"][method]]
        data_metric_wass_rel[method] = [(d-data_wass_original) for d in data["metric_wasserstein_scaled"][method]]
        data_metric_energy_rel[method] = [(d-data_energy_original) for d in data["metric_energy_scaled"][method]]

# Préparer les données pour le data_to_plotFrame
    

        data_to_plot['energy_distance'].extend(data_metric_energy[method])
        data_to_plot['wasserstein_distance'].extend(data_metric_wass[method])

        data_to_plot['privacy'].extend(data_privacy[method])
        data_to_plot['Category'].extend([method] * len(data_privacy[method]))

        data_to_plot_rel['wasserstein_distance'].extend(data_metric_wass_rel[method])
        data_to_plot_rel['energy_distance'].extend(data_metric_energy_rel[method])
        data_to_plot_rel['privacy'].extend(data_privacy_rel[method])
        data_to_plot_rel['Category'].extend([method] * len(data_privacy_rel[method]))
       
    data_to_plot['energy_distance'].extend(data_original["metric_energy_scaled"])
    data_to_plot['wasserstein_distance'].extend(data_original["metric_wasserstein_scaled"])

    data_to_plot['privacy'].extend(data_original["metric_nndr"])
    data_to_plot['Category'].extend(['original'] *100)

    
    df = pd.DataFrame(data_to_plot)

    df_rel = pd.DataFrame(data_to_plot_rel)
   

# Créer le graphique
    import itertools

# Définir des markers distincts (tu peux en ajouter d’autres si nécessaire)
    marker_cycle = itertools.cycle(['o', 's', 'D', '^', 'v', '<', '>', 'P', 'X', '*', 'H'])
    method_markers = {method: next(marker_cycle) for method in df['Category'].unique()}

# Crée la figure
    fig, ax = plt.subplots(1, 2, figsize=(15, 6))

# Couleur de l'original
    method_colors['original'] = '#000000'

# Palette et markers à appliquer
    palette = {method: color for method, color in method_colors.items() if method in df['Category'].unique()}
    markers = {method: marker for method, marker in method_markers.items() if method in df['Category'].unique()}

# Tracer avec la palette + markers personnalisés
    sns.scatterplot(data=df, x='energy_distance', y='privacy', hue='Category',
                style='Category', palette=palette, markers=markers,
                s=60, alpha=1, ax=ax[0])

    ax[0].tick_params(axis='x', labelsize=32)
    ax[0].tick_params(axis='y', labelsize=32)

    sns.scatterplot(data=df, x='wasserstein_distance', y='privacy', hue='Category',
                style='Category', palette=palette, markers=markers,
                s=60, alpha=1, ax=ax[1])

    ax[1].tick_params(axis='x', labelsize=32)
    ax[1].tick_params(axis='y', labelsize=32)

    ax[0].grid(True, linestyle='--', alpha=0.5)
    ax[1].grid(True, linestyle='--', alpha=0.5)
    ax[0].legend(title="Method", fontsize=18, title_fontsize=22, loc="lower right", frameon=True, markerscale=3)
    ax[1].legend(title="Method", fontsize=18, title_fontsize=22, loc="lower right", frameon=True, markerscale=3)
# Ajouter les médianes + barres d'erreur
    for method in METHODS:
        x_full = data_metric_wass[method]
        y_full = data_privacy[method]
        x = np.median(x_full)
        y = np.median(y_full)
        x_error = ([abs(x - np.quantile(x_full, 0.25))], [abs(x - np.quantile(x_full, 0.75))])
        y_error = ([abs(y - np.quantile(y_full, 0.25))], [abs(y - np.quantile(y_full, 0.75))])
    
        ax[1].errorbar(
        x=x, y=y, xerr=x_error, yerr=y_error,
        fmt=method_markers[method], linewidth=5, capsize=6,
        color=method_colors[method], markersize=6
        )


# Ajouter des indicateurs de dispersion

    #for _, row in df.iterrows():
    #    plt.plot([row['utility'] - row['X_Dispersion'], row['utility'] + row['X_Dispersion']],
    #         [row['privacy'], row['privacy']], color='gray', alpha=0.8)
    #    plt.plot([row['utility'], row['utility']],
    #         [row['privacy'] - row['Y_Dispersion'], row['privacy'] + row['Y_Dispersion']], color='gray', alpha=0.8)

# Ajouter des labels et un titre
    plt.title('Bivariate Plot with Dispersion Indicators')
    plt.ylabel('privacy')


    plt.subplots_adjust(right=0.8)

    im_path = output_path+"/"+dataset_name+"_metric_vs_privacy.pdf"

    plt.tight_layout() 
    plt.savefig(im_path, bbox_inches='tight')
    
    print(f"Saved: {output_path}")
    plt.close()




    fig, ax = plt.subplots(1, 1, figsize=(20, 15))



# Tracer avec la palette + markers personnalisés
    sns.scatterplot(data=df_rel, x='wasserstein_distance', y='privacy', hue='Category',
                style='Category', palette=palette, markers=markers,
                s=60, alpha=1, ax=ax)

    ax.tick_params(axis='x', labelsize=32)
    ax.tick_params(axis='y', labelsize=32)

    
    
# Ajouter les médianes + barres d'erreur
    for method in METHODS:
        x_full = data_metric_wass_rel[method]
        y_full = data_privacy_rel[method]
        x = np.median(x_full)
        y = np.median(y_full)
        x_error = ([abs(x - np.quantile(x_full, 0.25))], [abs(x - np.quantile(x_full, 0.75))])
        y_error = ([abs(y - np.quantile(y_full, 0.25))], [abs(y - np.quantile(y_full, 0.75))])
    
        ax.errorbar(
        x=x, y=y, xerr=x_error, yerr=y_error,
        fmt=method_markers[method], linewidth=5, capsize=6,
        color=method_colors[method], markersize=6
        )


# Ajouter des indicateurs de dispersion

    #for _, row in df.iterrows():
    #    plt.plot([row['utility'] - row['X_Dispersion'], row['utility'] + row['X_Dispersion']],
    #         [row['privacy'], row['privacy']], color='gray', alpha=0.8)
    #    plt.plot([row['utility'], row['utility']],
    #         [row['privacy'] - row['Y_Dispersion'], row['privacy'] + row['Y_Dispersion']], color='gray', alpha=0.8)

# Ajouter des labels et un titre
    ax.legend(title="Method", fontsize=30, title_fontsize=35, loc="lower right", frameon=True)

    plt.title('Utility vs Privacy on '+ dataset_name +' dataset', fontsize=30)
    plt.ylabel(r'Privacy : $ \frac{NNDR_{5 \%}(X_{true}, X_{synth})}{\hat{\mathbb{E}}_{X \sim F_{true}}(NNDR_{5 \%}(X_{true}, X))}$', fontsize=40)
    plt.xlabel(r'Utility : $d_{wasserstein}(X_{true}, X_{synth}) - \hat{\mathbb{E}}_{X \sim F_{true}}(d_{wasserstein}(X_{true}, X))$', fontsize=40)
    ax.grid(True, linestyle='--', alpha=0.5)

    plt.subplots_adjust(right=0.8)

    im_path = output_path+"/"+dataset_name+"rel_metric_vs_privacy.pdf"

    plt.tight_layout() 
    plt.savefig(im_path, bbox_inches='tight')
    
    plt.close()









    fig, ax = plt.subplots(1, 1, figsize=(20, 15))



# Tracer avec la palette + markers personnalisés
    sns.scatterplot(data=df_rel, x='energy_distance', y='privacy', hue='Category',
                style='Category', palette=palette, markers=markers,
                s=60, alpha=1, ax=ax)
    ax.tick_params(axis='x', labelsize=32)
    ax.tick_params(axis='y', labelsize=32)


# Ajouter les médianes + barres d'erreur
    for method in METHODS:
        x_full = data_metric_energy_rel[method]
        y_full = data_privacy_rel[method]
        x = np.median(x_full)
        y = np.median(y_full)
        x_error = ([abs(x - np.quantile(x_full, 0.25))], [abs(x - np.quantile(x_full, 0.75))])
        y_error = ([abs(y - np.quantile(y_full, 0.25))], [abs(y - np.quantile(y_full, 0.75))])
    
        ax.errorbar(
        x=x, y=y, xerr=x_error, yerr=y_error,
        fmt=method_markers[method], linewidth=5, capsize=6,
        color=method_colors[method], markersize=6
        )


# Ajouter des indicateurs de dispersion

    #for _, row in df.iterrows():
    #    plt.plot([row['utility'] - row['X_Dispersion'], row['utility'] + row['X_Dispersion']],
    #         [row['privacy'], row['privacy']], color='gray', alpha=0.8)
    #    plt.plot([row['utility'], row['utility']],
    #         [row['privacy'] - row['Y_Dispersion'], row['privacy'] + row['Y_Dispersion']], color='gray', alpha=0.8)

# Ajouter des labels et un titre
    plt.title('Utility vs Privacy on '+ dataset_name +' dataset', fontsize=30)
    plt.ylabel(r'Privacy : $ \frac{NNDR_{5 \%}(X_{true}, X_{synth})}{\hat{\mathbb{E}}_{X \sim F_{true}}(NNDR_{5 \%}(X_{true}, X))}$', fontsize=40)
    plt.xlabel(r'Utility : $d_{energy}(X_{true}, X_{synth}) - \hat{\mathbb{E}}_{X \sim F_{true}}(d_{energy}(X_{true}, X))$', fontsize=40)
    ax.grid(True, linestyle='--', alpha=0.5)    
    ax.legend(title="Method", fontsize=30, title_fontsize=35, loc="lower right", frameon=True, markerscale=3)
    plt.subplots_adjust(right=0.8)

    im_path = output_path+"/"+dataset_name+"rel_metric_energy_vs_privacy.pdf"

    plt.tight_layout() 
    plt.savefig(im_path, bbox_inches='tight')
    
    plt.close()


# Usage
if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python plot_privacy_from_pickle.py path/to/file.pkl [dataset_label]")
    else:
        pkl_file = sys.argv[1]
        pkl_path_original = sys.argv[2]
        output_path = sys.argv[3]
        dataset_name = sys.argv[4] if len(sys.argv) > 3 else ""
        plot_from_pickle(pkl_path=pkl_file, output_path=output_path, dataset_name=dataset_name)


        