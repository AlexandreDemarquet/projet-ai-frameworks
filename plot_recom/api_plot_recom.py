from flask import Flask, request,jsonify
from annoy import AnnoyIndex
import torch
import pandas as pd
import base64
from sklearn.feature_extraction.text import TfidfVectorizer
from transformers import DistilBertTokenizerFast


tfidf = TfidfVectorizer(stop_words='english')
distilbert_tokenizer = DistilBertTokenizerFast.from_pretrained('distilbert-base-uncased')


# Charger l'index
dimension = 576  # Dimension des embeddings
index_bow = AnnoyIndex(dimension, 'angular')
index_bow.load("annoy_index_bow.ann")  # Charger l'index pré-construit

index_distil= AnnoyIndex(dimension, 'angular')
index_distil.load("annoy_index_distil.ann")  # Charger l'index pré-construit

df = pd.read_csv("annoy-database.csv")

app = Flask(__name__)


df = pd.read_csv('annoy-database.csv')
paths_list = df['poster_path'].tolist()

def search(index,query_vector, k=5):
    indices = index.get_nns_by_vector(query_vector, k)
    paths = [paths_list[idx] for idx in indices]
    return paths


@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()

        plot = data.get("plot")
        model_type = data.get("model")

        if model_type == "bow":
            model=tfidf
            index=index_bow
        else:
            model=distilbert_tokenizer
            index=index_distil


        with torch.no_grad():
            embeddings = model(plot)

        query_vector = embeddings.flatten()

        # Récupère les chemins vers les images similaires
        image_paths = search(index,query_vector) 

        # Charger et encoder les images en base64
        encoded_images = []
        for path in image_paths:
            with open(path, "rb") as f:
                img_data = f.read()
                encoded = base64.b64encode(img_data).decode("utf-8")
                encoded_images.append(encoded)

        return jsonify({
            "images": encoded_images,
            "format": "base64",
            "note": "Chaque image est encodée en base64, à décoder côté client pour affichage"
        })

    except Exception as e:
        return jsonify({"error": str(e)})


if __name__ == "__main__":
    app.run(port=8080, debug=True, host="0.0.0.0")


