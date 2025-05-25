import streamlit as st
import requests
import io
import base64
from PIL import Image

DB_URL = "http://db:8000"

def find_similar(image):
    try:
        img_binary = io.BytesIO()
        image.save(img_binary, format="PNG")
        img_binary.seek(0)

        response = requests.post(
            f"{DB_URL}/predict",
            data=img_binary.getvalue(),
            headers={"Content-Type": "application/octet-stream"}
        )

        if response.status_code == 200:
            data = response.json()
            return data["images"]
        else:
            st.error(f"Erreur API: {response.status_code} - {response.text}")
            return []
    except Exception as e:
        st.error(f"Erreur de communication avec l'API: {str(e)}")
        return []

st.set_page_config(page_title="Reco Films", layout="centered")
st.title("🎬 Recommandations de Films à partir d'une Affiche")

uploaded_image = st.file_uploader("Uploader une image de poster de film", type=["png", "jpg", "jpeg"])

if uploaded_image:
    image = Image.open(uploaded_image)
    st.image(image, caption="🎞️ Image chargée", use_container_width=True)

    if st.button("🔍 Trouver des films similaires"):
        similar_images_b64 = find_similar(image)

        if similar_images_b64:
            st.subheader("🎯 Films similaires trouvés :")
            for i, b64_img in enumerate(similar_images_b64):
                try:
                    img_bytes = base64.b64decode(b64_img)
                    img = Image.open(io.BytesIO(img_bytes))
                    st.image(img, caption=f"Recommandation #{i+1}", use_container_width=True)
                except Exception as e:
                    st.write(f"Erreur lors de l'affichage d'une image : {str(e)}")
        else:
            st.warning("❌ Aucune recommandation trouvée.")
