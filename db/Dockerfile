# Utiliser une image Python
FROM python:3.10

# Définir le répertoire de travail
WORKDIR /app

# Copier les fichiers nécessaires
COPY . /app

# Installer les dépendances
RUN pip install fastapi uvicorn annoy numpy

# Exposer le port 8000
EXPOSE 8000

# Commande pour démarrer l'API
CMD ["uvicorn", "api_db:app", "--host", "0.0.0.0", "--port", "8000"]
