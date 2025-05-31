FROM nvidia/cuda:12.1.1-runtime-ubuntu22.04

WORKDIR /app

COPY requirements.txt /app

RUN apt-get update && \
    apt-get install -y python3 python3-pip curl && \
    pip3 install --no-cache-dir -r requirements.txt && \
    pip3 install gdown


RUN mkdir -p /weights

RUN gdown "https://drive.google.com/uc?export=download&id=1rJ8TqbqQvk7uQnmOh9mJbk_m4anlh8f-" -O /weights/filmClassifier.pth

# téléchargement db bag of word et distilbert
RUN gdown "https://drive.google.com/uc?export=download&id=11M0MCSJS2N_xBnVyklprZ8QHDERVL8mX" -O annoy_index_bow.ann
RUN gdown "https://drive.google.com/uc?export=download&id=1n7sIQLyznjvjXVrRkNe3bEI7PAoody2v" -O annoy_index_distil.ann

#Copy puis download pour permettre de garder en cache lorsque qu'on modifie juste un fichier source de l'api
COPY download_data.py /app
RUN python3 download_data.py


COPY . /app

EXPOSE 5000 

#CMD ["python3", "app.py"]
#CMD ["gunicorn", "-b", "0.0.0.0:5000", "api:app"]
CMD ["python3", "api.py"]

