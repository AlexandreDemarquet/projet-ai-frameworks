import gradio as gr
from PIL import Image
import requests
import io

# Définir les URLs des APIs
GENRE_API_URL = "http://127.0.0.1:5000/predict"
SIMILAR_API_URL = "http://127.0.0.1:5001/predict"

# Mapping des indices vers les noms des genres
GENRE_LABELS = {
    0: "Action",
    1: "Animation",
    2: "Comedy",
    3: "Documentary",
    4: "Drama",
    5: "Fantasy",
    6: "Horror",
    7: "Romance",
    8: "Science Fiction",
    9: "Thriller"
}

def recognize_genre(image):
    try:
        image = Image.fromarray(image.astype('uint8'))
        img_binary = io.BytesIO()
        image.save(img_binary, format="PNG")

        response = requests.post(GENRE_API_URL, data=img_binary.getvalue())

        if response.status_code == 200:
            predicted_index = response.json().get("prediction", -1)
            predicted_label = GENRE_LABELS.get(predicted_index, "Genre inconnu")
        else:
            predicted_label = "Erreur API"
        
        return predicted_label
    
    except Exception as e:
        return f"Erreur: {str(e)}"

def find_similar(image):
    try:
        image = Image.fromarray(image.astype('uint8'))
        img_binary = io.BytesIO()
        image.save(img_binary, format="PNG")

        # Simuler l'embedding (remplacer par un vrai modèle)
        embedding = b'fake_embedding_data'

        response = requests.post(SIMILAR_API_URL, data=embedding)

        if response.status_code == 200:
            images = [Image.open(io.BytesIO(x)) for x in response.content]
        else:
            images = []
        
        return images

    except Exception as e:
        return f"Erreur: {str(e)}"

def all(image):
    return recognize_genre(image), find_similar(image)

if __name__ == '__main__':
    gr.Interface(
        fn=all, 
        inputs="image", 
        outputs=["text", "image"],
        description="Mettre un poster de film pour prédire son genre et avoir des recommandations"
    ).launch(debug=True, share=True, server_name="0.0.0.0", server_port=7860)
