import os
import zipfile
import gdown

# URL du fichier Google Drive
DRIVE_URL = "https://drive.google.com/uc?export=download&id=1-1OSGlN2EOqyZuehBgpgI8FNOtK-caYf"  # ID extrait de l'URL

# Chemin du dossier où les images seront stockées
DATA_DIR = "data"
ZIP_PATH = "data/posters.zip"

# Créer le dossier si il n'existe pas
os.makedirs("data", exist_ok=True)

# Si les images ne sont pas encore présentes, les télécharger depuis Google Drive
if not os.path.exists(DATA_DIR):
    print("📦 Téléchargement des images depuis Google Drive...")
    
    # Télécharger le fichier depuis Google Drive
    gdown.download(DRIVE_URL, ZIP_PATH, quiet=False)

    # Extraction du fichier ZIP
    with zipfile.ZipFile(ZIP_PATH, "r") as zip_ref:
        zip_ref.extractall(DATA_DIR)
    print("✅ Données téléchargées et extraites.")
else:
    print("✔️ Les images sont déjà présentes.")
