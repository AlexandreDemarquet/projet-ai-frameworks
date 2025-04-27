import streamlit as st
import requests
import io
from PIL import Image

API_URL = "http://db:8000"


def find_similar(image):
    try:
        img_binary = io.BytesIO()
        image.save(img_binary, format="PNG")
        img_binary.seek(0)

        response = requests.post(
            f"{API_URL}/predict",
            data=img_binary.getvalue(),
            headers={"Content-Type": "application/octet-stream"}
        )

        if response.status_code == 200:
            paths = response.json()["prediction"]
            return paths
        else:
            st.error(f"Erreur API: {response.status_code} - {response.text}")
            return []
    except Exception as e:
        st.error(f"Erreur de communication avec l'API: {str(e)}")
        return []

st.title("🎬 Recommandations de Films à partir d'une Image")

uploaded_image = st.file_uploader("Uploader une image de poster de film", type=["png", "jpg", "jpeg"])

if uploaded_image:
    image = Image.open(uploaded_image)
    st.image(image, caption="Image chargée", use_column_width=True)

    if st.button("🔍 Trouver des films similaires"):
        similar_paths = find_similar(image)
        if similar_paths:
            st.write("Films similaires trouvés :")
            for path in similar_paths:
                try:
                    # Ici, on suppose que le chemin est accessible localement
                    img = Image.open(path)
                    st.image(img, caption=path, use_column_width=True)
                except Exception as e:
                    st.write(f"Impossible de charger {path}: {str(e)}")
        else:
            st.write("❌ Aucune recommandation trouvée.")
