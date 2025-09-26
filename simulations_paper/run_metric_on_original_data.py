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
from scipy.stats import energy_distance
from geomloss import SamplesLoss

# Ajouter le répertoire racine au chemin
sys.path.append(str(Path(__file__).resolve().parent.parent))


from tools.utils import bootstrap
from src.synthetic_gen import Generator
from test_on_simulated_dataset.generator_data.simulator_data_low_correlated import generate_lowcorr_data
from test_on_simulated_dataset.generator_data.simulator_data_high_correlated import generate_hightcorr_data
from test_on_simulated_dataset.generator_data.simulator_nls import generate_nonlinear_data

from src.mixture_gen import Mixture
from src.statis_gen import Statis
from metrics.utility_metrics import wasserstein_dist, energy, p_mse
from metrics.privacy_metrics import NNDR
from sklearn.datasets import load_breast_cancer


# Chemins
BASE_DIR = Path("/home/mfbeclin/synthetic_data/synthetic_data/synthetic_proj_python")
OUT_PATH = BASE_DIR / "simulations_paper/res_metrics/with_synthpop_cart"


# Régression de référence
METHODS = ['lowcorr', "highcorr", "nls"]
   

def run_experiment(method: str = "lowcorr", method_option: str = "best", output_prefix: str = "", with_synpop_cart = True):
    n = 500
    n_synth = 100

    if method == "lowcorr":
        DATA_PATH = BASE_DIR / "test_on_simulated_dataset/data_simulated/simulated_data_lowcorr.csv"
    elif method == "highcorr":
        DATA_PATH = BASE_DIR / "test_on_simulated_dataset/data_simulated/simulated_data_highcorr.csv"
    elif method == "nls":
        DATA_PATH = BASE_DIR / "test_on_simulated_dataset/data_simulated/simulated_data_nls.csv"
    elif method == "nls_5000":
        DATA_PATH = BASE_DIR / "test_on_simulated_dataset/data_simulated/simulated_data_nls_5000.csv"
    elif method == "nls_200":
        DATA_PATH = BASE_DIR / "test_on_simulated_dataset/data_simulated/simulated_data_nls_200.csv"
    else:
        raise ValueError(f"Invalid method: {method}")

    if method in ['nls_200', 'nls_5000']:
        df_true = pd.read_csv(DATA_PATH)
    else :
        df_true = pd.read_csv(DATA_PATH, index_col=0)
    

    # --- Méthodes synthétiques --- 
# Initialisation des structures
    def init_metrics():
        return []

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

    def set_global_seed(seed):
        np.random.seed(seed)
        random.seed(seed)
        os.environ["PYTHONHASHSEED"] = str(seed)    
        
# Générateur synthétique
    def generate_synth(method, seed):
        if method =="lowcorr":
            return generate_lowcorr_data(seed=seed)
        if method =="highcorr":
            return generate_hightcorr_data(seed=seed)
        if method in ["nls", "nls_200", "nls_5000"]:
            return generate_nonlinear_data(seed=seed)     

# Évaluation des métriques
    def evaluate_metrics(method, df_synth):
        metric_energy.append(energy(df_synth, df_true, scale=True))
        metric_wasserstein.append(wasserstein_dist(df_synth, df_true, scale=True))
        metric_energy_scaled.append(energy(df_synth, df_true, scale=True))
        metric_wasserstein_scaled.append(wasserstein_dist(df_synth, df_true, scale=True))
        metric_pmse_rf.append(p_mse(df_synth, df_true))
        metric_pmse_log.append(p_mse(df_true, df_synth, method='log'))

        dcr, nndr = NNDR(df_synth, df_true)
        metric_nndr.append(nndr)
        metric_dcr.append(dcr)

        dcr_all, nndr_all = NNDR(df_synth, df_true, all=True)
        metric_nndr_all.append(nndr_all)
        metric_dcr_all.append(dcr_all)


# Boucle principale
    for i in range(n_synth):
        #seed = randint(0, 99999)
        seed = secrets.randbelow(10**5)
        print(seed)
        set_global_seed(seed)

       
        df_synth = generate_synth(method, seed)
        print(df_synth)
        evaluate_metrics(method, df_synth)

    common_data = {
        "metric_energy": metric_energy,
        "metric_wasserstein": metric_wasserstein,
        "metric_energy_scaled": metric_energy_scaled,
        "metric_wasserstein_scaled": metric_wasserstein_scaled,
        "metric_pmse_rf": metric_pmse_rf,
        "metric_pmse_log": metric_pmse_log,
        "metric_nndr": metric_nndr,
        "metric_dcr": metric_dcr,
        "metric_nndr_all": metric_nndr_all,
        "metric_dcr_all": metric_dcr_all}
           

# Sauvegarde
    file_name = OUT_PATH / f"{method}_original.pkl"
    with open(file_name, "wb") as f:
        pickle.dump(common_data, f)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--data", choices=["lowcorr", "highcorr", "nls", "nls_200", "nls_5000"], required=True)
    args = parser.parse_args()
    run_experiment(method=args.data)