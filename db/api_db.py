from fastapi import FastAPI
from annoy import AnnoyIndex
import numpy as np

# Charger l'index
dimension = 100
index = AnnoyIndex(dimension, 'angular')
index.load("annoy_index.ann")  # Charger l'index pré-construit

app = FastAPI()

@app.get("/search/")
def search(vector: str, k: int = 5):
    """
    Recherche les k voisins les plus proches d'un vecteur donné.
    Le vecteur est passé sous forme de string séparée par des virgules.
    """
    try:
        query_vector = np.array([float(x) for x in vector.split(",")], dtype=np.float32)
        neighbors = index.get_nns_by_vector(query_vector, k, include_distances=True)
        return {"neighbors": neighbors}
    except Exception as e:
        return {"error": str(e)}

@app.get("/")
def root():
    return {"message": "API Annoy en ligne 🚀"}
