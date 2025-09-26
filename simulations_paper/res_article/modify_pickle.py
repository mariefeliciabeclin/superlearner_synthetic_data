import pickle
import pandas as pd
# --- Charger le pickle ---



file="/home/marie-felicia/synthetic_data/synthetic_proj_python/simulations_paper/res_article/all_datasets/breast_cancer.pkl"
file_tabgan = "/home/marie-felicia/synthetic_data/synthetic_proj_python/simulations_paper/res_article/with_tabgan/breast_cancer_best_n=100.pkl"
with open(file, "rb") as f:
    data = pickle.load(f)

with open(file_tabgan, "rb") as f:
    data_tabgan = pickle.load(f)

print(data_tabgan.keys())

# --- Modifier les données ---
# Supposons que c'est un dictionnaire

data["metric_energy"]["tabgan"] = data_tabgan["metric_energy"]["tabgan"]
data["metric_wasserstein"]["tabgan"] = data_tabgan["metric_wasserstein"]["tabgan"]

data["metric_energy_scaled"]["tabgan"] = data_tabgan["metric_energy_scaled"]["tabgan"]
data["metric_wasserstein_scaled"]["tabgan"] = data_tabgan["metric_wasserstein_scaled"]["tabgan"]

data["metric_dcr_all"]["tabgan"] = data_tabgan["metric_dcr_all"]["tabgan"]
data["metric_nndr_all"]["tabgan"] = data_tabgan["metric_nndr_all"]["tabgan"]

data["metric_dcr"]["tabgan"] = data_tabgan["metric_dcr"]["tabgan"]
data["metric_nndr"]["tabgan"] = data_tabgan["metric_nndr"]["tabgan"]

data["count_energy"]["tabgan"] = data_tabgan["count_energy"]["tabgan"]
data["count_wasserstein"]["tabgan"] = data_tabgan["count_wasserstein"]["tabgan"]

data["count_nndr"]["tabgan"] = data_tabgan["count_nndr"]["tabgan"]

data["count_pmse"]["tabgan"] = data_tabgan["count_pmse"]["tabgan"]

# --- Sauvegarder à nouveau ---
with open(file, "wb") as f:
    pickle.dump(data, f)

print("Fichier pickle mis à jour ✅")