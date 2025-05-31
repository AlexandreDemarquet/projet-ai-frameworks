import os
import shutil
import kagglehub

os.makedirs("data", exist_ok=True)

path = kagglehub.dataset_download("rounakbanik/the-movies-dataset")

# Copier les fichiers dans le dossier ./data/
for file_name in os.listdir(path):
    full_file_path = os.path.join(path, file_name)
    if os.path.isfile(full_file_path):
        shutil.copy(full_file_path, "data")

print("Fichiers copiés dans le dossier ./data/")


