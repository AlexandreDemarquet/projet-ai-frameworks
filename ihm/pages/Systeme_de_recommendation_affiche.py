import streamlit as st
import requests
import io
from PIL import Image

def find_similar(image):
    try:
        img_binary = io.BytesIO()
        image.save(img_binary, format="PNG")
        embedding = b'fake_embedding_data'  # Simuler l'embedding
        response = requests.post(st.session_state["SIMILAR_API_URL"], data=embedding)
        
        if response.status_code == 200:
            return [Image.open(io.BytesIO(x)) for x in response.content]
        else:
            return []
    except Exception as e:
        return f"Erreur: {str(e)}"

st.title("Recommandations de Films à partir d'une image")
uploaded_image = st.file_uploader("Uploader une image de poster de film", type=["png", "jpg", "jpeg"])

if uploaded_image:
    image = Image.open(uploaded_image)
    st.image(image, caption="Image chargée", use_column_width=True)
    

    if st.button("Trouver des films similaires"):
        similar_images = find_similar(image)
        if isinstance(similar_images, list) and similar_images:
            st.write("Films similaires :")
            for img in similar_images:
                st.image(img, use_column_width=True)
        else:
            st.write("Aucune recommandation trouvée.")