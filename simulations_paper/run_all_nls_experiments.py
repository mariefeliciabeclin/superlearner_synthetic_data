import subprocess
from pathlib import Path


# Paramètres
data_types = ["nls_500","nls_200","nls_5000" ]
method_options = ["best"]

n_synth = 100 # Doit correspondre à celui utilisé dans run_experiment.py

# Chemin vers les fichiers .pkl
BASE_DIR = Path("/home/mfbeclin/synthetic_data/synthetic_data/synthetic_proj_python")
OUT_PATH = BASE_DIR / "simulations_paper/res_metrics/nls_version"
OUT_PATH_IMAGE = BASE_DIR / "simulations_paper/res_image/nls_version"

def main(synthpopcart: bool):
# Lancer les expériences
    for data in data_types:
        for method in method_options:
            filename = f"{data}_{method}_n={n_synth}.pkl"
            pkl_path = OUT_PATH / filename

            if not pkl_path.exists():
                print(f"Running experiment for data={data}, method={method}")
                print(synthpopcart)
                if synthpopcart : 
                    subprocess.run([
                        "python", "/home/mfbeclin/synthetic_data/synthetic_data/synthetic_proj_python/simulations_paper/run_experiment.py",
                        "--data", data,
                        "--methods", method,
                        "--synthpopcart"
                     ])
                else : 
                    subprocess.run([
                        "python", "/home/mfbeclin/synthetic_data/synthetic_data/synthetic_proj_python/simulations_paper/run_experiment.py",
                        "--data", data,
                        "--methods", method,
                     ])

            else:
                print(f"Skipping: {filename} already exists")

        # Plot metrics
            subprocess.run(["python", "/home/mfbeclin/synthetic_data/synthetic_data/synthetic_proj_python/simulations_paper/plot_metrics_from_pickle.py", pkl_path, data , OUT_PATH_IMAGE, ])

        # Plot privacy
            subprocess.run(["python", "/home/mfbeclin/synthetic_data/synthetic_data/synthetic_proj_python/simulations_paper/plot_privacy_from_pickle.py", pkl_path, OUT_PATH_IMAGE,])


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="let's do xp")
    parser.add_argument('--synthpopcart', action='store_true')
    parser.add_argument('--no_wass', action='store_true')
    

    args = parser.parse_args()

    main(synthpopcart=args.synthpopcart)