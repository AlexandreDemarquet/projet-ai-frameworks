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
    
def get_smoothgrad_map(image):
    try:
        img_binary = io.BytesIO()
        image.save(img_binary, format="PNG")
        # files = {'image': img_binary}
        response = requests.post(st.session_state["SMOOTHGRAD_API_URL"], data=img_binary.getvalue())
        
        if response.status_code == 200:
            return Image.open(io.BytesIO(response.content))
        else:
            return None
    except Exception as e:
        return f"Erreur: {str(e)}"

def get_lime_map(image):
    try:
        img_binary = io.BytesIO()
        image.save(img_binary, format="PNG")
        # files = {'image': img_binary}
        response = requests.post(st.session_state["LIME_API_URL"], data=img_binary.getvalue())
        
        if response.status_code == 200:
            return Image.open(io.BytesIO(response.content))
        else:
            return None
    except Exception as e:
        return f"Erreur: {str(e)}"

def get_shap_map(image):
    try:
        img_binary = io.BytesIO()
        image.save(img_binary, format="PNG")
        url = st.session_state["SHAP_API_URL"]
        
        response = requests.post(url, data=img_binary.getvalue())
        
        if response.status_code == 200:
            return Image.open(io.BytesIO(response.content))
        else:
            return None
    except Exception as e:
        return None




st.title("Prédiction de Genre et Interprétabilité")
uploaded_image = st.file_uploader("Uploader une image de poster de film", type=["png", "jpg", "jpeg"])

if uploaded_image:
    image = Image.open(uploaded_image)
    st.image(image, caption="Image chargée", use_container_width=True)
    
    if st.button("Prédire le genre"):
        genre = recognize_genre(image)
        st.write(f"Genre prédit : {genre}")

    if st.button("Afficher la smooth grad map"):
        saliency_map = get_smoothgrad_map(image)
        if saliency_map:
            st.image(saliency_map, caption="Carte de saillance (saliency map)", use_container_width=True)
        else:
            st.write("Impossible de générer la smooth grad map.")
    if st.button("Afficher la lime map"):
        saliency_map = get_lime_map(image)
        if saliency_map:
            st.image(saliency_map, caption="Carte de saillance (saliency map)", use_container_width=True)
        else:
            st.write("Impossible de générer la lime map.")
    if st.button("Afficher la SHAP map"):
        shap_map = get_shap_map(image)
        if shap_map:
            st.image(shap_map, caption="Carte SHAP (importance par pixel)", use_container_width=True)
        else:
            st.write("❌ Impossible de générer la carte SHAP.")
