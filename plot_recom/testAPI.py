import requests

# URL de base de l'API
API_URL = "http://127.0.0.1:8080"
TEST_PLOT = "Un homme fait un tour du monde"


def test_predict():
    try:
        json_data_all = []
        for model_type in ["bow", "distil"]:
            response = requests.post(
                f"{API_URL}/predict",
                json={"plot": TEST_PLOT, "model": model_type},
                headers={"Content-Type": "application/json"}
            )

            print("[INFO] Status Code:", response.status_code)
            print("[INFO] Raw Response Text:", response.text)

            # Assurer que la réponse est correcte (200 OK)
            response.raise_for_status()  # Lève une erreur explicite si code != 200

            # Parsing la réponse JSON
            json_data = response.json()
            print("[INFO] Parsed JSON:", json_data)

            # Vérifier que la clé 'prediction' existe dans la réponse
            assert "titles" in json_data, "'titles' not in response JSON"
            assert isinstance(json_data["titles"], list), "'titles' is not a list"

            # Vérifier que les éléments de la liste sont des chaînes ou des entiers
            for item in json_data["titles"]:
                assert isinstance(item, (str, int)), f"Item {item} is not a string or int"

            # Vérifier que la clé 'prediction' existe dans la réponse
            assert "plots" in json_data, "'plots' not in response JSON"
            assert isinstance(json_data["plots"], list), "'plots' is not a list"

            # Vérifier que les éléments de la liste sont des chaînes ou des entiers
            for item in json_data["plots"]:
                assert isinstance(item, (str, int)), f"Item {item} is not a string or int"

            print("✅ Test passed! Results:", json_data["titles"])
            json_data_all.append(json_data)
        return json_data_all

    except Exception as e:
        print("❌ Exception occurred during test:")
        print(e)

if __name__ == "__main__":
    response = test_predict()
