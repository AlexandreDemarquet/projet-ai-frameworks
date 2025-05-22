# Utiliser une image Python
FROM python:3.10

# Définir le répertoire de travail
WORKDIR /app

# Copier le fichier requirements.txt dans le container
COPY requirements.txt /app

# Mettre à jour et installer les dépendances
RUN apt update && apt install -y python3 python3-pip && \
    pip3 install --no-cache-dir -r requirements.txt

# Copier tous les fichiers nécessaires
COPY . /app
RUN python download_data.py


# Télécharger les données et le modèle depuis Google Drive
RUN gdown "https://drive.google.com/uc?export=download&id=11M0MCSJS2N_xBnVyklprZ8QHDERVL8mX" -O annoy_index_bow.ann
RUN gdown "https://drive.google.com/uc?export=download&id=1n7sIQLyznjvjXVrRkNe3bEI7PAoody2v" -O annoy_index_distil.ann


# Exposer le port 8080 pour l'API
EXPOSE 8080

# Commande pour démarrer l'API avec Gunicorn
CMD ["gunicorn", "-b", "0.0.0.0:8080", "api_plot_recom:app"]
