FROM python:3.10

WORKDIR /app

RUN pip install annoy

COPY test_annoy.py . 
CMD ["tail", "-f", "/dev/null"]  # Pour garder le conteneur actif
