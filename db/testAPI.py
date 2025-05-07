import requests
import numpy as np
import matplotlib.pyplot as plt

# URL de base de l'API
API_URL = "http://127.0.0.1:8000"
TEST_IMAGE_PATH = "/home/tristan/Images/Captures d’écran/Capture d’écran du 2024-11-25 09-13-28.png"
TEST_IMAGE_PATH = "sorted_movie_posters_paligema/animation/158.jpg"


def test_predict():
    try:
        print("[INFO] Opening test image:", TEST_IMAGE_PATH)
        with open(TEST_IMAGE_PATH, "rb") as img:
            img_data = img.read()
            print("[INFO] Sending POST request...")
            response = requests.post(
                f"{API_URL}/predict",
                data=img_data,
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
        assert "prediction" in json_data, "'prediction' not in response JSON"
        assert isinstance(json_data["prediction"], list), "'prediction' is not a list"

        # Vérifier que les éléments de la liste sont des chaînes ou des entiers
        for item in json_data["prediction"]:
            assert isinstance(item, (str, int)), f"Item {item} is not a string or int"

        print("✅ Test passed! Results:", json_data["prediction"])
        return json_data

    except Exception as e:
        print("❌ Exception occurred during test:")
        print(e)

if __name__ == "__main__":
    response = test_predict()
