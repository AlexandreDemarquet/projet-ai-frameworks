import streamlit as st
import requests
import io
from PIL import Image

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
        response = requests.post(st.session_state["GENRE_API_URL"], data=img_binary.getvalue())
        
        if response.status_code == 200:
            predicted_index = response.json().get("prediction", -1)
            return GENRE_LABELS.get(predicted_index, "Genre inconnu")
        else:
            return "Erreur API"
    except Exception as e:
        return f"Erreur: {str(e)}"


st.title("Prédiction de Genre ")
uploaded_image = st.file_uploader("Uploader une image de poster de film", type=["png", "jpg", "jpeg"])

if uploaded_image:
    image = Image.open(uploaded_image)
    st.image(image, caption="Image chargée", use_column_width=True)
    
    if st.button("Prédire le genre"):
        genre = recognize_genre(image)
        st.write(f"Genre prédit : {genre}")
    