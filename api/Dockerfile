FROM nvidia/cuda:12.1.1-runtime-ubuntu22.04

WORKDIR /app

COPY requirements.txt /app

RUN apt update && apt install -y python3 python3-pip python3-pip curl && \
    pip3 install --no-cache-dir -r requirements.txt

COPY . /app


# Créer le répertoire pour les poids
RUN mkdir -p /app/weights
RUN ls -l ./

# Télécharger les poids du modèle depuis Google Drive
RUN curl -L -o weights/filmClassifier.pth "https://drive.google.com/uc?export=download&id=1rJ8TqbqQvk7uQnmOh9mJbk_m4anlh8f"


EXPOSE 5000
CMD ["python3", "app.py"]
