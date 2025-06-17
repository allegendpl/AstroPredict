# app.py

import streamlit as st
import pandas as pd
from data_loader import load_data
from model_trainer import train_models, split_data, cross_validate_models, save_models
from evaluation import evaluate_all, save_report
from visualization import plot_feature_distribution, plot_confusion_matrix, plot_roc_curve

# Page setup
st.set_page_config(page_title="AstroPredict - Solar Flare AI Predictor", layout="wide")

st.title(" AstroPredict — AI Solar Flare Prediction Tool")

# Sidebar
st.sidebar.header("Options")
data_option = st.sidebar.selectbox("Select Data Source", ("Real Solar Flare Data", "Fake Generated Data"))
action = st.sidebar.radio("Choose an action:", ("View Dataset", "Train Models", "Evaluate Models", "Make Prediction"))

@st.cache_data
def load_selected_data(option):
    if option == "Real Solar Flare Data":
        df = load_data(real_file='data/solar_flare_real.csv')
    else:
        from src.data_loader import generate_fake_data
        df = generate_fake_data()
    return df

df = load_selected_data(data_option)

if action == "View Dataset":
    st.subheader("Dataset Preview")
    st.dataframe(df.head(20))
    st.subheader("Feature Distributions")
    fig = plot_feature_distribution(df)
    st.pyplot(fig)

elif action == "Train Models":
    st.subheader("Training Machine Learning Models")
    X_train, X_test, y_train, y_test = split_data(df)
    models = train_models(X_train, y_train)
    cv_scores = cross_validate_models(models, X_train, y_train)
    st.write("### Cross-Validation Accuracy Scores:")
    for name, score in cv_scores.items():
        st.write(f"- **{name}**: {score:.4f}")
    save_models(models)
    st.success(" Models trained and saved!")

elif action == "Evaluate Models":
    st.subheader("Model Evaluation on Test Set")
    X_train, X_test, y_train, y_test = split_data(df)
    from src.model_trainer import load_models
    # Load saved models
    import pickle, os
    model_dir = 'models/trained_models'
    models = {}
    for model_file in os.listdir(model_dir):
        if model_file.endswith('.pkl'):
            name = model_file.replace('.pkl','')
            with open(os.path.join(model_dir, model_file), 'rb') as f:
                models[name] = pickle.load(f)
    results = evaluate_all(models, X_test, y_test)
    save_report(results)

    # Visualize evaluation results
    model_choice = st.selectbox("Select Model to Visualize", list(models.keys()))
    if model_choice:
        st.write(f"### Confusion Matrix - {model_choice}")
        fig_cm = plot_confusion_matrix(results[model_choice]['confusion_matrix'], model_choice)
        st.pyplot(fig_cm)
        st.write(f"### ROC Curve - {model_choice}")
        fig_roc = plot_roc_curve(models[model_choice], X_test, y_test, model_choice)
        st.pyplot(fig_roc)

elif action == "Make Prediction":
    st.subheader("Predict Solar Flare Occurrence")

    # Load best model (choose RandomForest as default)
    model_path = 'models/trained_models/RandomForestClassifier.pkl'
    try:
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
    except FileNotFoundError:
        st.error("Please train the models first!")
        st.stop()

    # Input features
    sunspot_num = st.number_input("Sunspot Number", min_value=0, max_value=1000, value=50)
    radio_flux = st.number_input("Radio Flux", min_value=0.0, max_value=500.0, value=100.0)
    xray_emission = st.number_input("X-Ray Emission", min_value=0.0, max_value=20.0, value=5.0)

    input_data = pd.DataFrame({
        "Sunspot Number": [sunspot_num],
        "Radio Flux": [radio_flux],
        "X-Ray Emission": [xray_emission]
    })

    if st.button("Predict Flare Probability"):
        proba = model.predict_proba(input_data)[0][1]
        st.write(f" Probability of Solar Flare: **{proba*100:.2f}%**")
        if proba > 0.5:
            st.warning(" High risk of solar flare!")
        else:
            st.success(" Low risk of solar flare.")

# Footer
st.markdown("---")
st.markdown("**AstroPredict** by Allegendpl  | Powered by Stellar Gateway Hackathon")
