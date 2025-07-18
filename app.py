import streamlit as st
import pandas as pd
import numpy as np
import joblib
import google.generativeai as genai

# Load model and encoder
model = joblib.load("risk_model.pkl")

# Gemini setup
genai.configure(api_key="AIzaSyCKHagVbXd4MF9YBg_f788OelNubmLT72w")  # Replace with your API key
gemini_model = genai.GenerativeModel(model_name="models/gemini-2.5-pro")

st.title("Food Insecurity Risk Predictor")

region = st.selectbox("Select Region", ["North", "South", "East", "West"])
income = st.number_input("Household Income (₹)", min_value=0, value=5000)
family_size = st.slider("Family Size", 1, 15, 5)
price_increase = st.checkbox("Has food price increased recently?")
market_distance = st.slider("Market Distance (km)", 1, 50, 10)
water_quality = st.selectbox("Water Quality", ["Good", "Poor"])

if st.button("Predict Risk"):
    region_encoded = {"East": 0, "North": 1, "South": 2, "West": 3}[region]
    water_quality_val = 1 if water_quality == "Good" else 0

    input_data = pd.DataFrame([[
        region_encoded,
        income,
        family_size,
        int(price_increase),
        market_distance,
        water_quality_val
    ]], columns=["region_encoded", "income", "family_size", "price_increase", "market_distance", "water_quality"])

    prediction = model.predict(input_data)[0]
    base_recommendation = "Deploy emergency food aid." if prediction == 1 else "Continue monitoring."

    prompt = f"""The household is from {region} with an income of ₹{income}, family size of {family_size}, 
    water quality: {water_quality}, market distance: {market_distance} km.
    Generate a short action plan for an NGO to support this household."""

    # st.info("Generating AI-based action plan...")
    # response = gemini_model.generate_content(prompt)
    # ai_plan = response.text.strip()
    with st.spinner("Generating AI-based action plan..."):
    response = gemini_model.generate_content(prompt)
    ai_plan = response.text.strip()


    st.subheader("Prediction Result:")
    st.write("🔴 High Risk" if prediction == 1 else "🟢 Low Risk")
    st.write("💡 Recommendation:", base_recommendation)
    st.write("📋 AI Action Plan:")
    st.markdown(ai_plan)

