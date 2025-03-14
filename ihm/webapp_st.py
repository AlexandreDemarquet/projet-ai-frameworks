import streamlit as st
import requests
import io
from PIL import Image

# Définir les URLs des APIs
GENRE_API_URL = "http://api:5000/predict"
SIMILAR_API_URL = "http://api:5001/predict"

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
        img_binary = io.BytesIO()
        image.save(img_binary, format="PNG")
        response = requests.post(GENRE_API_URL, data=img_binary.getvalue())
        
        if response.status_code == 200:
            predicted_index = response.json().get("prediction", -1)
            return GENRE_LABELS.get(predicted_index, "Genre inconnu")
        else:
            return "Erreur API"
    except Exception as e:
        return f"Erreur: {str(e)}"

def find_similar(image):
    try:
        img_binary = io.BytesIO()
        image.save(img_binary, format="PNG")
        embedding = b'fake_embedding_data'  # Simuler l'embedding
        response = requests.post(SIMILAR_API_URL, data=embedding)
        
        if response.status_code == 200:
            return [Image.open(io.BytesIO(x)) for x in response.content]
        else:
            return []
    except Exception as e:
        return f"Erreur: {str(e)}"

st.title("Prédiction de Genre et Recommandations de Films")
uploaded_image = st.file_uploader("Uploader une image de poster de film", type=["png", "jpg", "jpeg"])

if uploaded_image:
    image = Image.open(uploaded_image)
    st.image(image, caption="Image chargée", use_column_width=True)
    
    if st.button("Prédire le genre"):
        genre = recognize_genre(image)
        st.write(f"Genre prédit : {genre}")
    
    if st.button("Trouver des films similaires"):
        similar_images = find_similar(image)
        if isinstance(similar_images, list) and similar_images:
            st.write("Films similaires :")
            for img in similar_images:
                st.image(img, use_column_width=True)
        else:
            st.write("Aucune recommandation trouvée.")
