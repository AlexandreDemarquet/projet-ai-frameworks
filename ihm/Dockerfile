# FROM python:3.9-slim

# WORKDIR /app

# RUN apt-get update && apt-get install -y \
#     build-essential \
#     curl \
#     software-properties-common \
#     git \
#     && rm -rf /var/lib/apt/lists/*

# COPY . /app
    
# RUN pip3 install -r requirements.txt

# EXPOSE 7860

# HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health

# # ENTRYPOINT ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
# ENTRYPOINT ["python", "webapp.py"]

# Utiliser une image Python comme base
FROM python:3.9

# Définir le répertoire de travail
WORKDIR /app

# Copier les fichiers de l'application
COPY . /app

# Installer les dépendances
RUN pip install --no-cache-dir -r requirements.txt

# Exposer le port 5001 pour Streamlit
EXPOSE 7860

# Commande pour exécuter l'application Streamlit
CMD ["streamlit", "run", "webapp_st.py", "--server.port=7860", "--server.address=0.0.0.0"]
