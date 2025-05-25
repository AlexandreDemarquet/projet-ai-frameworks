import streamlit as st
st.set_page_config(layout="wide")


st.session_state["GENRE_API_URL"] = "http://api:5000/predict"
st.session_state["SIMILAR_API_URL"] = "http://api:5000/predict_genre"
st.session_state["SMOOTHGRAD_API_URL"] = "http://api:5000/smoothgrad"
st.session_state["LIME_API_URL"] = "http://api:5000/lime"
st.session_state["PLOT_API_URL"] = "http://api:5000/predict_text"


st.title("Projet IA Frameworks")
st.write("2025 - Dion Thomas / Gay Tristan / Gris Clément / Demarquet Alexandre")



st.divider()
page_prediction_genre, page_sys_reco_affiche, page_sys_reco_plot = st.columns(3,border=True)

with page_prediction_genre:
    st.write("Prédiction du genre d'une affiche de film")
    if st.button("Accès à la page de prédiction de genre"):
        st.switch_page("./pages/Prediction_de_genre.py")
with page_sys_reco_affiche:
    st.write("Système de recommandation basé sur des affiches de films")
    if st.button("Accès à la page de recommandation d'affiche"):
        st.switch_page("./pages/Systeme_de_recommendation_affiche.py")
with page_sys_reco_plot:
    st.write("Système de recommandation basé sur des plots de films")
    if st.button("Accès à la page de recommandation de plot"):
        st.switch_page("./pages/Systeme_de_recommendation_plot.py")