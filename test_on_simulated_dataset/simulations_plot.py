import subprocess
import sys
import os

def execute_python_file(file_path, data_arg, all_arg):
    try:
        args = [sys.executable, file_path, "--data", data_arg]
        if all_arg:
            args.append("--all")  # flag booléen

        result = subprocess.run(args, capture_output=True, text=True)
        print(f"Output of {file_path}:")
        print(result.stdout)
        print(f"Errors of {file_path}:")
        print(result.stderr)
    except Exception as e:
        print(f"An error occurred while executing {file_path}: {e}")

if __name__ == "__main__":
    current_directory = os.path.dirname(os.path.abspath(__file__))
    #files_to_execute = ["test_mixture_best_hightcorr.py", "test_mixture_best_lowcorr.py","test_mixture_all_hightcorr.py", "test_mixture_all_lowcorr.py","test_statis_dual_best_hightcorr.py", "test_statis_dual_best_lowcorr.py","test_statis_dual_all_hightcorr.py", "test_statis_dual_all_lowcorr.py",
    # "test_statis_double_best_hightcorr.py", "test_statis_double_best_lowcorr.py","test_statis_double_all_hightcorr.py", "test_statis_double_all_lowcorr.py"]    # Spécifie ici les fichiers à exécuter
    #files_to_execute =  ["plot_methods_all_hightcorr.py","plot_methods_best_hightcorr.py", "plot_methods_all_lowcorr.py", "plot_methods_best_lowcorr.py"]    # Spécifie ici les fichiers à exécuter
    files_to_execute = ["plot_methods.py"]
    for file in files_to_execute:
        print('execute : '+file)
        file_path = os.path.join(current_directory, file)  # Crée le chemin absolu
        if os.path.exists(file_path):  # Vérifie si le fichier existe
            execute_python_file(file_path, data_arg="hightcorr", all_arg=True)
            execute_python_file(file_path, data_arg="lowcorr", all_arg=True)
            execute_python_file(file_path, data_arg="nonlinear", all_arg=True)
            execute_python_file(file_path, data_arg="nls", all_arg=True)

            execute_python_file(file_path, data_arg="hightcorr", all_arg=False)
            execute_python_file(file_path, data_arg="lowcorr", all_arg=False)
            execute_python_file(file_path, data_arg="nonlinear", all_arg=False)
            execute_python_file(file_path, data_arg="nls", all_arg=False)
         
        else:
            print(f"File not found: {file_path}")

