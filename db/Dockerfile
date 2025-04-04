# Utiliser une image Python
FROM python:3.10

# Définir le répertoire de travail
WORKDIR /app


COPY requirements.txt /app
# Installer les dépendances
RUN apt update && apt install -y python3 python3-pip && \
    pip3 install --no-cache-dir -r requirements.txt

# Copier les fichiers nécessaires
COPY . /app

# Exposer le port 8000
EXPOSE 8000

# Commande pour démarrer l'API avec Gunicorn
CMD ["gunicorn", "-b", "0.0.0.0:8000", "api_db:app"]
