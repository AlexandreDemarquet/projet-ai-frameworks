import requests
import os

API_URL = "http://localhost:5000"
TEST_IMAGE_PATH = "content/sorted_movie_posters_paligema/action/110.jpg"

def test_predict():
    with open(TEST_IMAGE_PATH, "rb") as img:
        files = {"file": img}
        response = requests.post(f"{API_URL}/predict", data=img.read(), headers={"Content-Type": "application/octet-stream"})
    
    assert response.status_code == 200
    json_data = response.json()
    assert "prediction" in json_data
    assert isinstance(json_data["prediction"], int)


if __name__ == "__main__":
    test_predict()
    print("All tests passed!")
