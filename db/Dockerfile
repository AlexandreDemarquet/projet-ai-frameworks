# Utiliser une image Python
FROM python:3.10

# Définir le répertoire de travail
WORKDIR /app

# Copier le fichier requirements.txt dans le container
COPY requirements.txt /app

# Mettre à jour et installer les dépendances
RUN apt update && apt install -y python3 python3-pip && \
    pip3 install --no-cache-dir -r requirements.txt

# Installer gdown pour télécharger depuis Google Drive
RUN pip install gdown

# Copier tous les fichiers nécessaires
COPY . /app

# Télécharger les données et le modèle depuis Google Drive
#RUN python /app/download_data.py
RUN python download_data.py && python buildIndexDb.py

# Exposer le port 8000 pour l'API
EXPOSE 8000

# Commande pour démarrer l'API avec Gunicorn
CMD ["gunicorn", "-b", "0.0.0.0:8000", "api_db:app"]
