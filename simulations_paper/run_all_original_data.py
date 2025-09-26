import subprocess
from pathlib import Path


# Paramètres
data_types = ["lowcorr", "highcorr", "nls", "breast_cancer"]
method_options = ["best"]

n_synth = 100 # Doit correspondre à celui utilisé dans run_experiment.py

# Chemin vers les fichiers .pkl
BASE_DIR = Path("/home/mfbeclin/synthetic_data/synthetic_data/synthetic_proj_python")
OUT_PATH = BASE_DIR / "simulations_paper/res_metrics"


data_types = ['nls_200', 'nls_5000']


def main():
# Lancer les expériences
    for data in data_types:
        print(data)
        subprocess.run([
            "python", "/home/mfbeclin/synthetic_data/synthetic_data/synthetic_proj_python/simulations_paper/run_metric_on_original_data.py",
            "--data", data])

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="let's do xp")
    args = parser.parse_args()

    main()
   

