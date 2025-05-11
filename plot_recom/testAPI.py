import requests
import numpy as np
import matplotlib.pyplot as plt

# URL de base de l'API
API_URL = "http://127.0.0.1:8080"
TEST_PLOT = "Un homme fait un tour du monde"


def test_predict():
    try:
        for model_type in ["bow", "distil"]:
            response = requests.post(
                f"{API_URL}/predict",
                json={"plot": TEST_PLOT, "model": model_type},
                headers={"Content-Type": "application/octet-stream"}
            )

            print("[INFO] Status Code:", response.status_code)
            print("[INFO] Raw Response Text:", response.text)

            # Assurer que la réponse est correcte (200 OK)
            response.raise_for_status()  # Lève une erreur explicite si code != 200

            # Parsing la réponse JSON
            json_data = response.json()
            print("[INFO] Parsed JSON:", json_data)

            # Vérifier que la clé 'prediction' existe dans la réponse
            assert "images" in json_data, "'prediction' not in response JSON"
            assert isinstance(json_data["images"], list), "'prediction' is not a list"

            # Vérifier que les éléments de la liste sont des chaînes ou des entiers
            for item in json_data["images"]:
                assert isinstance(item, (str, int)), f"Item {item} is not a string or int"

            print("✅ Test passed! Results:", json_data["images"])
            return json_data

    except Exception as e:
        print("❌ Exception occurred during test:")
        print(e)

if __name__ == "__main__":
    response = test_predict()
