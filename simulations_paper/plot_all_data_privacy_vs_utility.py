import subprocess
from pathlib import Path


# Paramètres
#dataset_name = ['lowcorr', "highcorr", "nls"]
#dataset_file = ["lowcorr_best_n=100.pkl", "highcorr_best_n=100.pkl", "nls_best_n=100.pkl"]
#dataset_original = ["lowcorr_original.pkl", "highcorr_original.pkl", "nls_original.pkl"]  # <- corriger ici aussi


dataset_name = ['nls_200', "nls_500", "nls_5000"]
dataset_file = ["nls_200_n=100.pkl", "nls_500_n=100.pkl", "nls_5000_n=100.pkl"]
dataset_original = ["nls_200_original.pkl", "nls_500_original.pkl", "nls_5000_original.pkl"]  # <- corriger ici aussi





# Chemins
BASE_DIR = Path("/home/mfbeclin/synthetic_data/synthetic_data/synthetic_proj_python")
OUT_PATH = BASE_DIR / "simulations_paper/res_metrics"
OUT_PATH_IM = BASE_DIR / "simulations_paper/res_image"
DATA_PATH = BASE_DIR / "simulations_paper/res_metrics/with_synthpop_cart"
DATA_PATH_ORIGINAL = BASE_DIR / "simulations_paper/res_metrics"


#Chemin : nls version
BASE_DIR = Path("/home/mfbeclin/synthetic_data/synthetic_data/synthetic_proj_python")
OUT_PATH = BASE_DIR / "/simulations_paper/res_article/nls_version"
OUT_PATH_IM = BASE_DIR / "simulations_paper/res_article/nls_version/images"
DATA_PATH = BASE_DIR / "simulations_paper/res_article/nls_version"
DATA_PATH_ORIGINAL = BASE_DIR / "simulations_paper/res_article/nls_version"




def main():
    # Lancer les expériences
    for i, data in enumerate(dataset_name):
        result = subprocess.run([
            "python",
            str(BASE_DIR / "simulations_paper/plot_privacy_vs_metric.py"),
            str(DATA_PATH / dataset_file[i]),
            str(DATA_PATH_ORIGINAL / dataset_original[i]),
            str(OUT_PATH_IM),
            data
        ])
        print(f"Process finished with return code {result.returncode}")

if __name__ == "__main__":
    main()
