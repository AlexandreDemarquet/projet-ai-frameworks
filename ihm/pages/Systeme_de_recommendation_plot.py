import streamlit as st
import requests


API_URL = st.session_state["PLOT_API_URL"]

def find_similar(plot,model_type):
    try:

        response = requests.post(
            f"{API_URL}/predict",
            json={"plot": plot, "model": model_type},
            headers={"Content-Type": "application/json"}
        )

        if response.status_code == 200:
            data = response.json()
            return data["titles"],data["plots"]
        else:
            st.error(f"Erreur API: {response.status_code} - {response.text}")
            return [],[]
    except Exception as e:
        st.error(f"Erreur de communication avec l'API: {str(e)}")
        return [],[]

# Interface utilisateur Streamlit
st.set_page_config(page_title="Reco Films", layout="centered")
st.title("🎬 Recommandations de Films à partir d'un résumé")

user_input = st.text_area("✍️ Entrer une description ou un résumé de film")

model_type = st.radio("🧠 Choisir le modèle :", ["Bag of word", "DistilBert"])
model_type = {"Bag of word": "bow", "DistilBert": "distil"}[model_type]


if user_input:
    st.write("📝 Texte saisi :", user_input)

    if st.button("🔍 Trouver des films similaires"):
        similar_titles,similar_plots = find_similar(user_input,model_type)

        if similar_titles:
            st.subheader("🎯 Films similaires recommandés :")

            for i, (title, plot) in enumerate(zip(similar_titles, similar_plots), start=1):
                st.markdown(
                    f"""
                    <div style="
                        background-color: #2c2f38;
                        padding: 15px;
                        border-radius: 10px;
                        margin-bottom: 15px;
                        color: #e0e0e0;
                        ">
                        <h4 style="margin-top: 0;">#{i} 🎬 {title}</h4>
                        <p style="margin-bottom: 0;">{plot}</p>
                    </div>
                    """,
                    unsafe_allow_html=True
                )
        else:
            st.warning("❌ Aucune recommandation trouvée.")