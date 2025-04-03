import requests
import numpy as np

# URL de base de l'API
API_URL = "http://127.0.0.1:8000"
TEST_IMAGE_PATH = "/home/tristan/Images/Captures d’écran/Capture d’écran du 2024-11-25 09-13-28.png"

def test_predict():
    with open(TEST_IMAGE_PATH, "rb") as img:
        files = {"file": img}
        response = requests.post(f"{API_URL}/predict", data=img.read(), headers={"Content-Type": "application/octet-stream"})
    
    assert response.status_code == 200
    json_data = response.json()
    assert "prediction" in json_data
    assert isinstance(json_data["prediction"], list)


if __name__ == "__main__":
    test_predict()
    print("All tests passed!")




# # Tester si l'API est en ligne
# def test_api_status():
#     response = requests.get(BASE_URL)
#     print("Statut de l'API:", response.json())

# # Tester la recherche d'un vecteur aléatoire
# def test_annoy_search():
#     # Générer un vecteur aléatoire de 100 dimensions
#     img_pil = Image.open(io.BytesIO(img_binary))

#         # Transform the PIL image
#     tensor = transform(img_pil).to(device)
#     tensor = tensor.unsqueeze(0)

#     with torch.no_grad():
#         embeddings = model(tensor)

#     query_vector = embeddings.cpu().numpy()
#     vector_str = ",".join(map(str, vector))  # Convertir en string pour l'API

#     # Envoyer la requête
#     response = requests.get(f"{BASE_URL}/search/", params={"vector": vector_str, "k": 5})

#     # Afficher la réponse
#     print("Résultat de la recherche:", response.json())

# if __name__ == "__main__":
#     test_api_status()
#     test_annoy_search()
