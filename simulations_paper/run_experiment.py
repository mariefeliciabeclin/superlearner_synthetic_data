import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
from pathlib import Path
from random import randint
import random

import pickle
import sys
import os
import secrets
from sklearn import preprocessing


# Ajouter le répertoire racine au chemin
sys.path.append(str(Path(__file__).resolve().parent.parent))
from nonlinear_regression.nls import regression_nls

from tools.utils import bootstrap
from src.synthetic_gen import Generator
from src.mixture_gen import Mixture
from src.statis_gen import Statis
from metrics.utility_metrics import wasserstein_dist, energy, p_mse
from metrics.privacy_metrics import NNDR
from sklearn.datasets import load_breast_cancer


# Chemins

#BASE_DIR = Path("/home/mfbeclin/synthetic_data/synthetic_data/synthetic_proj_python")
#OUT_PATH = BASE_DIR / "simulations_paper/res_metrics/res_final"

BASE_DIR = Path("/home/mfbeclin/synthetic_data/synthetic_data/synthetic_proj_python")
OUT_PATH = BASE_DIR / "simulations_paper/res_article/without_synthpopcart"
OUT_PATH_IMAGE = BASE_DIR / "simulations_paper/res_article/without_synthpopcart/images"

# Régression de référence



def run_experiment(data_type: str = "lowcorr", method_option: str = "best", output_prefix: str = "", with_synpop_cart = True):
    
    n_synth = 100

    # --- Chargement données ---
    if data_type == "lowcorr":
        DATA_PATH = BASE_DIR / "test_on_simulated_dataset/data_simulated/simulated_data_lowcorr.csv"
    elif data_type == "highcorr":
        DATA_PATH = BASE_DIR / "test_on_simulated_dataset/data_simulated/simulated_data_highcorr.csv"
    elif data_type == "nls_500":
        DATA_PATH = BASE_DIR / "test_on_simulated_dataset/data_simulated/simulated_data_nls.csv"
    elif data_type == "nls_5000":
        DATA_PATH = BASE_DIR / "test_on_simulated_dataset/data_simulated/simulated_data_nls_5000.csv"
    elif data_type == "nls_200":
        DATA_PATH = BASE_DIR / "test_on_simulated_dataset/data_simulated/simulated_data_nls_200.csv"
    elif data_type == "breast_cancer":
        data_bc = load_breast_cancer()
        df_true =pd.DataFrame(data_bc["data"])
    else:
        raise ValueError(f"Invalid data_type: {data_type}")

    if not data_type == "breast_cancer":
        if data_type in ['nls_200', 'nls_5000']:
            df_true = pd.read_csv(DATA_PATH)
        else :
            df_true = pd.read_csv(DATA_PATH, index_col=0)

    if method_option == "best":
        if with_synpop_cart :
            METHODS_SYNTHGEN = ['synthpop_norm', 'synthpop_cart', 'avatar_ncp=4']
            METHODS_LABS = ['synthpop\n norm', 'synthpop\ncart', 'avatar\n ncp=4']
        else : 
            METHODS_SYNTHGEN = ['synthpop_norm',  'avatar_ncp=4']
            METHODS_LABS = ['synthpop\n norm', 'avatar\n ncp=4']
    elif method_option == "all":
        METHODS_SYNTHGEN = ['synthpop_norm', 'synthpop_cart', 'avatar_ncp=4', "synthcity_ctgan", "sdv_ctgan",'tabgan' ]
        METHODS_LABS = ['synthpop\n norm', 'synthpop\ncart', 'avatar\n ncp=4', "synthcity\nctgan", "sdv\nctgan", 'tabgan']
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

    n = len(df_true)
    if data_type in ["lowcorr", "highcorr"]:
        var = ['X0', 'X1', 'X2', 'X3']
        reg_linear = sm.OLS(df_true['Y'], df_true[var]).fit()
        params_true = reg_linear.params
        conf_int = reg_linear.conf_int(alpha=0.05).T
        conf = [[conf_int[col][0] - params_true[col], conf_int[col][1] - params_true[col]] for col in var]
    if data_type in ["nls"]:
        var = ['X0', 'X1', 'X2', 'X3']
        params_true= regression_nls(df=df_true)
        

    def init_metrics():
        return {key: [] for key in METHODS}

    res = init_metrics()
    res_conf_size = init_metrics()
    metric_energy = init_metrics()
    metric_wasserstein = init_metrics()
    metric_energy_scaled = init_metrics()
    metric_wasserstein_scaled = init_metrics()
    metric_pmse_rf = init_metrics()
    metric_pmse_log = init_metrics()
    metric_nndr = init_metrics()
    metric_dcr = init_metrics()
    metric_nndr_all = init_metrics()
    metric_dcr_all = init_metrics()

    

    count_energy = dict.fromkeys(METHODS, 0)
    count_wasserstein = dict.fromkeys(METHODS, 0)
    count_pmse = dict.fromkeys(METHODS, 0)
    count_nndr = dict.fromkeys(METHODS, 0)

    def set_global_seed(seed):
        np.random.seed(seed)
        random.seed(seed)
        os.environ["PYTHONHASHSEED"] = str(seed)    
        

# Générateur synthétique
    def generate_synth(method, seed):
        if method.startswith("synthpop"):
            sub = method.split('_')[1]
            return Generator(method='synthpop', params_synthpop={'sub_method': [sub]}, seed=seed).get_generator()
        elif method.startswith("avatar"):
            ncp = int(method.split('=')[1])
            return Generator(method='avatar', params_avatar={'ncp': ncp}, seed=seed).get_generator()
        return Generator(method=method, seed=seed).get_generator()

# Évaluation des métriques
    def evaluate_metrics(method, df_synth):
        if data_type in ["lowcorr", "highcorr"]:
            reg = sm.OLS(df_synth['Y'], df_synth[var]).fit()
            params = pd.DataFrame(reg.params).T
            if params_true is not None:
                delta_params = (params - params_true)  # Résultat sous forme de DataFrame 1 ligne
                res[method].append(delta_params)
            
            conf_res = reg.conf_int(alpha=0.05).apply(lambda x: abs(x[1] - x[0]), axis=1).tolist()
            res_conf_size[method].append(conf_res)
        if data_type in ["nls", "nls_5000", "nls_200"]:
            try : 
                params =  regression_nls(df=df_synth)
            except : 
                params = ["fail", "fail", "fail", "fail", "fail"]
            res[method].append(params)
            #if params_true is not None:
                #delta_params = (params - params_true)  # Résultat sous forme de DataFrame 1 ligne
                #res[method].append(delta_params)
                
        metric_energy[method].append(energy(df_synth, df_true, scale=True))
        metric_wasserstein[method].append(wasserstein_dist(df_synth, df_true, scale=True))
        metric_energy_scaled[method].append(energy(df_synth, df_true, scale=True))
        metric_wasserstein_scaled[method].append(wasserstein_dist(df_synth, df_true, scale=True))
        metric_pmse_rf[method].append(p_mse(df_synth, df_true))
        metric_pmse_log[method].append(p_mse(df_true, df_synth, method='log'))

        dcr, nndr = NNDR(df_synth, df_true)
        metric_nndr[method].append(nndr)
        metric_dcr[method].append(dcr)

        dcr_all, nndr_all = NNDR(df_synth, df_true, all=True)
        metric_nndr_all[method].append(nndr_all)
        metric_dcr_all[method].append(dcr_all)


# Boucle principale
    for i in range(n_synth):
        #seed = randint(0, 99999)
        seed = secrets.randbelow(10**5)
        set_global_seed(seed)
        DF = []

        for method in METHODS_SYNTHGEN:
            gen_model = generate_synth(method, seed)
            gen_model.fit(df_true)
            print(df_true)
            print("on simule")
            if data_type == "breast_cancer":
                n= 569
            if data_type in ["lowcorr", "highcorr", "nls","nls_5000", "nls_200"]:
                var = ['X0', 'X1', 'X2', 'X3']
                df_synth = pd.DataFrame(gen_model.generate(n), columns=var + ['Y'])
                print(df_synth)
            else: 
                df_synth = pd.DataFrame(gen_model.generate(n))
            DF.append(df_synth)
            print("on mesure")
            evaluate_metrics(method, df_synth)

    # Agrégation mixture
        for w_method in ["wasserstein","energy", ]:
            key = f"mixture_{w_method}"
            df_mix = Mixture(weight_method=w_method).gen(DF=DF, df_true=df_true)
            evaluate_metrics(key, df_mix)

    # Agrégation statis
        for key, params in statis_param.items():
            if data_type in ["lowcorr", "highcorr", "nls","nls_5000", "nls_200"]:
                df_stat = pd.DataFrame(Statis(**params).gen(DF=DF, df_true=df_true), columns=var + ['Y'])
            else : 
                 df_stat = pd.DataFrame(Statis(**params).gen(DF=DF, df_true=df_true))
            evaluate_metrics(key, df_stat)

    # Comparaison pour les scores
        def best_method(metric_dict, mode="min"):
            scores = [metric_dict[m][i] for m in METHODS]
            idx = np.argmin(scores) if mode == "min" else np.argmax(scores)
            return METHODS[idx]

        count_energy[best_method(metric_energy_scaled)] += 1
        count_wasserstein[best_method(metric_wasserstein_scaled)] += 1
        count_nndr[best_method(metric_nndr, mode="max")] += 1
        count_pmse[best_method(metric_pmse_log)] += 1


    common_data = {
        "METHODS": METHODS,
        "METHODS_LABS": METHODS_LABS,
        "metric_energy": metric_energy,
        "metric_wasserstein": metric_wasserstein,
        "metric_energy_scaled": metric_energy_scaled,
        "metric_wasserstein_scaled": metric_wasserstein_scaled,
        "metric_pmse_rf": metric_pmse_rf,
        "metric_pmse_log": metric_pmse_log,
        "metric_nndr": metric_nndr,
        "metric_dcr": metric_dcr,
        "metric_nndr_all": metric_nndr_all,
        "metric_dcr_all": metric_dcr_all,
        "count_energy": count_energy,
        "count_wasserstein": count_wasserstein,
        "count_nndr": count_nndr,
        "count_pmse": count_pmse,
        "res_conf_size" : res_conf_size
        }
    if data_type in ["lowcorr", "highcorr", "nls_500","nls_5000", "nls_200"]:
            common_data["res"] = res

# Sauvegarde
    if method_option == "all":
        file_name = OUT_PATH / f"{data_type}_{method_option}_n={n_synth}.pkl"
    else : 
        file_name = OUT_PATH / f"{data_type}_n={n_synth}.pkl"

    with open(file_name, "wb") as f:
        pickle.dump(common_data, f)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--data", choices=["lowcorr", "highcorr", "nls_500", "breast_cancer", "nls_5000", "nls_200"], required=True)
    parser.add_argument("--methods", choices=["best", "all"], default="best")
    parser.add_argument("--prefix", type=str, default="norm")
    parser.add_argument('--synthpopcart', action='store_true')

    args = parser.parse_args()

    run_experiment(data_type=args.data, method_option=args.methods, output_prefix=args.prefix, with_synpop_cart = args.synthpopcart)    # --- Load or segment ---
 
    