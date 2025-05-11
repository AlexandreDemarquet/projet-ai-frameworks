import os
import zipfile
import kagglehub

# Créer le dossier si il n'existe pas
os.makedirs("data", exist_ok=True)

# Download latest version
path = kagglehub.dataset_download("rounakbanik/the-movies-dataset")

print("Path to dataset files:", path)

