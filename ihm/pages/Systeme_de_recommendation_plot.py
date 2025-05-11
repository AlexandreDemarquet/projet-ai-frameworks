import streamlit as st
import requests
import io
import base64
from PIL import Image

# Adresse de l'API (assure-toi qu'elle est accessible depuis Streamlit)
API_URL = "http://db:8080"

def find_similar(plot,model_type):
    try:

        response = requests.post(
            f"{API_URL}/predict",
            json={"plot": plot, "model": model_type},
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

# Interface utilisateur Streamlit
st.set_page_config(page_title="Reco Films", layout="centered")
st.title("🎬 Recommandations de Films à partir d'une Affiche")

user_input = st.text_area("✍️ Entrer une description ou un résumé de film")

model_type = st.radio("🧠 Choisir le modèle :", ["bow", "distil"])

if user_input:
    st.write("📝 Texte saisi :", user_input)

    if st.button("🔍 Trouver des films similaires"):
        similar_images_b64 = find_similar(user_input,model_type)

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
