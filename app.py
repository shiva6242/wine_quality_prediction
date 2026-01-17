import streamlit as st
import pandas as pd
import joblib

# ---------------------------------
# Load saved objects
# ---------------------------------
model = joblib.load("models/wine_model.pkl")
scaler = joblib.load("models/scaler.pkl")
features = joblib.load("models/features.pkl")
feature_means = joblib.load("models/feature_means.pkl")

# Load dataset for preview
dataset = pd.read_csv("WineQT.csv")

# ---------------------------------
# Page Config
# ---------------------------------
st.set_page_config(
    page_title="Wine Quality Predictor",
    layout="centered"
)

st.title("🍷 Wine Quality Prediction App")
st.write(
    "Predict whether a wine is **Good** or **Bad** using only "
    "the most influential chemical properties."
)

st.markdown("---")

# ---------------------------------
# User Input Section
# ---------------------------------
st.subheader("🔢 Enter Wine Properties")

user_input = {}

for feature in features:
    user_input[feature] = st.number_input(
        label=feature.replace("_", " ").title(),
        value=float(feature_means[feature]),  # meaningful default
        step=0.1
    )

input_df = pd.DataFrame([user_input])

# ---------------------------------
# Prediction
# ---------------------------------
if st.button("Predict Wine Quality"):
    input_scaled = scaler.transform(input_df)
    prediction = model.predict(input_scaled)[0]
    probability = model.predict_proba(input_scaled)[0][1]

    if prediction == 1:
        st.success(
            f"🍷 **Good Quality Wine**\n\n"
            f"Confidence: {probability:.2f}"
        )
    else:
        st.error(
            f"❌ **Bad Quality Wine**\n\n"
            f"Confidence: {1 - probability:.2f}"
        )

st.markdown("---")

# ---------------------------------
# Dataset Preview Button
# ---------------------------------
if st.button("📄 Show 20 Rows of Dataset"):
    st.subheader("Wine Dataset (First 20 Rows)")
    st.dataframe(dataset.head(20))
