# -*- coding: utf-8 -*-
"""
Streamlit front‑end for the Sentiment API with loader spinner and user feedback.

Prerequisites:
    pip install streamlit requests

Run:
    STREAMLIT_API_URL=http://localhost:8000  streamlit run streamlit_app.py

Features
========
* Auto‑discovers available models via GET /models
* Text area for user input (one sentence per line)
* Select model (radio) + optional threshold slider
* Displays probability & label in a data‑frame‑like table
* Shows a loading spinner while waiting for API response
* Feedback buttons (Correct / Incorrect) to send user feedback via POST /feedback
"""
import os
import requests
import streamlit as st
import pandas as pd

# Config
STREAMLIT_API_URL = os.getenv("STREAMLIT_API_URL", "http://localhost:8000").rstrip('/')

st.set_page_config(page_title="Sentiment Analysis", layout="centered")
st.title("Sentiment Analysis App with Feedback")

# 1. Récupérer la liste des modèles disponibles
try:
    resp = requests.get(f"{STREAMLIT_API_URL}/models")
    resp.raise_for_status()
    models = resp.json().get("models", [])
except Exception as e:
    st.error(f"Impossible de récupérer la liste des modèles : {e}")
    st.stop()

if not models:
    st.error("Aucun modèle disponible depuis l'API.")
    st.stop()

# 2. Sélection du modèle
selected_model = st.radio("Choisissez un modèle", models)

# 3. Zone de texte pour saisir
user_input = st.text_area(
    "Entrez une ou plusieurs phrases (une par ligne)",
    height=150,
    placeholder="Ex: Je suis très content de ce produit\nEx: C'était une mauvaise expérience"
)

# 4. Slider pour ajuster le seuil (optionnel)
threshold = st.slider(
    "Seuil de décision",
    min_value=0.0,
    max_value=1.0,
    value=0.5,
    step=0.01
)

# Variables to hold last prediction
last_probs = []
last_labels = []

# 5. Bouton pour envoyer la requête de prédiction
if st.button("Predict"):
    if not user_input.strip():
        st.error("⚠️ Veuillez entrer au moins une phrase.")
    else:
        predict_url = f"{STREAMLIT_API_URL}/predict/"
        payload = {
            "text": user_input,
            "model_name": selected_model,
            "threshold": threshold
        }
        try:
            with st.spinner("Prédiction en cours, veuillez patienter..."):
                resp = requests.post(predict_url, json=payload, timeout=30)
                resp.raise_for_status()
                data = resp.json()

            # Stocker les résultats pour feedback
            last_probs = data.get("probabilities", [])
            last_labels = data.get("labels", [])

            # Construire un DataFrame pour l'affichage
            results = []
            for prob, lbl in zip(last_probs, last_labels):
                results.append({"probability": prob, "label": lbl})
            df = pd.DataFrame(results)

            st.subheader("Résultats de la prédiction")
            st.dataframe(df)

            # Afficher les boutons de feedback
            st.markdown("---")
            st.write("**Votre retour : la prédiction est-elle correcte ?**")
            col1, col2 = st.columns(2)
            with col1:
                if st.button("👍 Correct"):
                    feedback_payload = {
                        "text": user_input,
                        "model": selected_model,
                        "probability": results[0]['probability'],
                        "validated": True
                    }
                    try:
                        resp_fb = requests.post(f"{STREAMLIT_API_URL}/feedback", json=feedback_payload)
                        resp_fb.raise_for_status()
                        st.success("Merci pour votre retour ! 🙏")
                    except Exception as e:
                        st.error(f"Échec de l'envoi du feedback : {e}")
            with col2:
                if st.button("👎 Incorrect"):
                    feedback_payload = {
                        "text": user_input,
                        "model": selected_model,
                        "probability": results[0]['probability'],
                        "validated": False
                    }
                    try:
                        resp_fb = requests.post(f"{STREAMLIT_API_URL}/feedback", json=feedback_payload)
                        resp_fb.raise_for_status()
                        st.success("Merci pour votre retour ! 🙏")
                    except Exception as e:
                        st.error(f"Échec de l'envoi du feedback : {e}")

        except requests.exceptions.HTTPError as errh:
            st.error(f"Requête HTTP échouée : {errh}")
        except requests.exceptions.RequestException as err:
            st.error(f"Erreur réseau ou timeout : {err}")
        except Exception as e:
            st.error(f"Erreur inattendue : {e}")

# 6. Afficher l'URL de l'API pour debug
st.sidebar.markdown("---")
st.sidebar.write(f"API URL utilisée : {STREAMLIT_API_URL}")
