import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
from pathlib import Path
from random import randint
import pickle
import sys

from sklearn import preprocessing
from scipy.stats import energy_distance
from geomloss import SamplesLoss

# Ajouter le répertoire racine au chemin
sys.path.append(str(Path(__file__).resolve().parent.parent))


from tools.utils import bootstrap
from src.synthetic_gen import Generator
from src.mixture_gen import Mixture
from src.statis_gen import Statis
from metrics.utility_metrics import wasserstein_dist, energy, p_mse
from metrics.privacy_metrics import NNDR
from sklearn.datasets import load_breast_cancer
from sklearn.metrics import accuracy_score

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
# Chemins

BASE_DIR = Path("/home/mfbeclin/synthetic_data/synthetic_data/synthetic_proj_python")
OUT_PATH = BASE_DIR / "simulations_paper/res_article/classification"



# Régression de référence



def run_experiment( method_option: str = "best", output_prefix: str = "", with_synpop_cart = True):
    n_synth = 100


    
    data_bc = load_breast_cancer()
    df =pd.DataFrame(data_bc["data"])
    target = pd.DataFrame(data_bc["target"], columns=['target'])
  
    X_train, X_test, y_train, y_test = train_test_split(
    df, target, test_size=0.25, random_state=42)
    n= 426  

    clf_true_list= []
    for i in range(100):
        clf_true = RandomForestClassifier()
        clf_true.fit(X_train, y_train)

        y_pred_true = clf_true.predict(X_test)
        clf_true_accu = accuracy_score(y_test, y_pred_true)
        clf_true_list.append(clf_true_accu) 
    
    df_true_complete=pd.concat([X_train, y_train], axis=1)
    df_true = X_train

    df_true_complete.columns = df_true_complete.columns.astype(str)
    col_name = df_true_complete.columns
 
    # --- Méthodes synthétiques ---
    if method_option == "best":
        if with_synpop_cart :
            METHODS_SYNTHGEN = ['synthpop_norm', 'synthpop_cart', 'avatar_ncp=4']
            METHODS_LABS = ['synthpop\n norm', 'synthpop\ncart', 'avatar\n ncp=4']
        else : 
            METHODS_SYNTHGEN = ['synthpop_norm',  'avatar_ncp=4']
            METHODS_LABS = ['synthpop\n norm', 'avatar\n ncp=4']
    elif method_option == "all":
        METHODS_SYNTHGEN = ['synthpop_norm', 'synthpop_cart', 'avatar_ncp=4', "synthcity_ctgan", "sdv_ctgan"]
        METHODS_LABS = ['synthpop\n norm', 'synthpop\ncart', 'avatar\n ncp=4', "synthcity\nctgan", "sdv\nctgan"]
    else:
        raise ValueError(f"Invalid method_option: {method_option}")


    METHODS = METHODS_SYNTHGEN + [
        "mixture_wasserstein", "mixture_energy",
        "statis_dual_energy_eigen", "statis_dual_wass_eigen",
        "statis_dual_energy_weight", "statis_dual_wass_weight"
    ]
    METHODS_LABS =  METHODS_LABS + [
        "mixture\n wasserstein", "mixture\n energy",
        "statis dual \n energy_eigen", "statis dual \n wass eigen",
    "   statis_dual \n energy weight", "statis dual \n wass weight"
    ]
    statis_param = {
        "statis_dual_energy_eigen": {"statis_method": "dual", "weight_method": "energy", "compromis_method": "eigen", "delta_weight": True},
        "statis_dual_wass_eigen": {"statis_method": "dual", "weight_method": "wasserstein", "compromis_method": "eigen", "delta_weight": True},
        "statis_dual_energy_weight": {"statis_method": "dual", "weight_method": "energy", "compromis_method": "weight", "delta_weight": True},
        "statis_dual_wass_weight": {"statis_method": "dual", "weight_method": "wasserstein", "compromis_method": "weight", "delta_weight": True},
    }

# Initialisation des structures


    def init_metrics():
        return {key: [] for key in METHODS}

    res = init_metrics()
    metric_energy_scaled = init_metrics()
    metric_wasserstein_scaled = init_metrics()
    metric_nndr = init_metrics()
    metric_dcr = init_metrics()
    metric_nndr_all = init_metrics()
    metric_dcr_all = init_metrics()
    metric_acc_train_synth_test_real = init_metrics()
 
    
        

# Générateur synthétique
    def generate_synth(method, seed):
        if method.startswith("synthpop"):
            print(method)
            sub = method.split('_')[1]
            print(sub)
            if sub=='norm':
                return Generator(method='synthpop', params_synthpop={'sub_method': ["parametric"]}, seed=seed).get_generator()
            else : 
                print("ok")
                return Generator(method='synthpop', params_synthpop={'sub_method': ["cart"]}, seed=seed).get_generator()
        elif method.startswith("avatar"):
            ncp = int(method.split('=')[1])
            return Generator(method='avatar', params_avatar={'ncp': ncp}, seed=seed).get_generator()
        return Generator(method=method, seed=seed).get_generator()

# Évaluation des métriques
    def evaluate_metrics(method, df_synth):
       

        metric_energy_scaled[method].append(energy(df_synth, df_true_complete, scale=True))
        metric_wasserstein_scaled[method].append(wasserstein_dist(df_synth,df_true_complete, scale=True))
        
        dcr, nndr = NNDR(df_synth, df_true_complete)
        metric_nndr[method].append(nndr)
        metric_dcr[method].append(dcr)

        dcr_all, nndr_all = NNDR(df_synth, df_true_complete, all=True)
        metric_nndr_all[method].append(nndr_all)
        metric_dcr_all[method].append(dcr_all)

        clf = RandomForestClassifier()

        clf.fit(df_synth.drop(columns='target'), df_synth['target'])
        y_pred_synth = clf.predict(X_test)
        clf_accu = accuracy_score(y_test, y_pred_synth)
        metric_acc_train_synth_test_real[method].append(clf_accu)





# Boucle principale
    for i in range(n_synth):
        seed = randint(0, 99999)
        DF = []

        for method in METHODS_SYNTHGEN:
            print(method)
            print(len)
            gen_model = generate_synth(method, seed)
            gen_model.fit(df_true_complete)
         
            df_synth = pd.DataFrame(gen_model.generate(n), columns = col_name)
            df_synth['target'] = (df_synth['target'] > 0.5).astype(int)

            DF.append(df_synth)
            evaluate_metrics(method, df_synth)
            

    # Agrégation mixture
        for w_method in ["wasserstein", "energy"]:
            key = f"mixture_{w_method}"
            df_mix = Mixture(weight_method=w_method).gen(DF=DF, df_true=df_true_complete)
            df_mix = pd.DataFrame(df_mix, columns = col_name)
            df_mix['target'] = (df_mix['target'] > 0.5).astype(int)
            evaluate_metrics(key, df_mix)

    # Agrégation statis
        for key, params in statis_param.items():
            df_stat = pd.DataFrame(Statis(**params).gen(DF=DF, df_true=df_true_complete), columns=col_name)
            df_stat['target'] = (df_stat['target'] > 0.5).astype(int)
            evaluate_metrics(key, df_stat)

    # Comparaison pour les scores
        def best_method(metric_dict, mode="min"):
            scores = [metric_dict[m][i] for m in METHODS]
            idx = np.argmin(scores) if mode == "min" else np.argmax(scores)
            return METHODS[idx]

      


    common_data = {
        "METHODS": METHODS,
        "METHODS_LABS": METHODS_LABS,
        "metric_energy_scaled": metric_energy_scaled,
        "metric_wasserstein_scaled": metric_wasserstein_scaled,
        "metric_nndr": metric_nndr,
        "metric_dcr": metric_dcr,
        "metric_nndr_all": metric_nndr_all,
        "metric_dcr_all": metric_dcr_all,
        "metric_acc_train_synth_test_real": metric_acc_train_synth_test_real,
        "metric_acc_train_real_test_real": clf_true_list
        }
    

# Sauvegarde
    file_name = OUT_PATH / f"bc_classif_n={n_synth}.pkl"
    with open(file_name, "wb") as f:
        pickle.dump(common_data, f)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--synthpopcart', action='store_true')

    args = parser.parse_args()

    run_experiment( with_synpop_cart = args.synthpopcart)