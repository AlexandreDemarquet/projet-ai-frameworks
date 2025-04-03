from flask import Flask, request,jsonify
from annoy import AnnoyIndex
import numpy as np
from embeggings_model import model
import torch
import torchvision.transforms as transforms
from PIL import Image
import io
import pandas as pd

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model=model.to(device)



# Charger l'index
dimension = 100
index = AnnoyIndex(dimension, 'angular')
index.load("annoy_index.ann")  # Charger l'index pré-construit
df = pd.read_csv("annoy-database.csv")

app = Flask(__name__)

mean = [ 0.485, 0.456, 0.406 ]
std = [ 0.229, 0.224, 0.225 ]
normalize = transforms.Normalize(mean, std)
inv_normalize = transforms.Normalize(
   mean= [-m/s for m, s in zip(mean, std)],
   std= [1/s for s in std]
)

transform = transforms.Compose([transforms.Resize((224, 224)),
                                transforms.ToTensor(),
                                normalize])

@app.route('/predict', methods=['POST'])
def predict():
    """
    Recherche les k voisins les plus proches d'une image.
    On utilise l'embedding de l'image calculé à partir de mobilnet.
    """
    try:
        img_binary = request.data
        img_pil = Image.open(io.BytesIO(img_binary))

        # Transform the PIL image
        tensor = transform(img_pil).to(device)
        tensor = tensor.unsqueeze(0)

        with torch.no_grad():
            embeddings = model(tensor)

        k=5
        query_vector = embeddings.cpu().numpy()
        neighbors = index.get_nns_by_vector(query_vector, k, include_distances=True)
        paths = df.iloc[neighbors]['path'].unique()
        return jsonify({"prediction": neighbors})
    except Exception as e:
        return {"error": str(e)}

if __name__ == "__main__":
    app.run(port=8000, debug=True, host="0.0.0.0")

#docker run -p 8000:8000 api-annoy

