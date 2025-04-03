import requests
import numpy as np

# URL de base de l'API
BASE_URL = "http://127.0.0.1:8000"

# Tester si l'API est en ligne
def test_api_status():
    response = requests.get(BASE_URL)
    print("Statut de l'API:", response.json())

# Tester la recherche d'un vecteur aléatoire
def test_annoy_search():
    # Générer un vecteur aléatoire de 100 dimensions
    vector = np.random.rand(100).astype(np.float32)
    vector_str = ",".join(map(str, vector))  # Convertir en string pour l'API

    # Envoyer la requête
    response = requests.get(f"{BASE_URL}/search/", params={"vector": vector_str, "k": 5})

    # Afficher la réponse
    print("Résultat de la recherche:", response.json())

if __name__ == "__main__":
    test_api_status()
    test_annoy_search()
