import streamlit as st
import pickle
import numpy as np

# Load model
model = pickle.load(open("model.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))

st.title("House Price Prediction")

# Inputs
area = st.number_input("Area (sq ft)")
bedrooms = st.number_input("Bedrooms", min_value=1)
bathrooms = st.number_input("Bathrooms", min_value=1)
stories = st.number_input("Stories", min_value=1)
parking = st.number_input("Parking Spaces", min_value=0)

if st.button("Predict Price"):

    data = np.array([[area, bedrooms, bathrooms, stories, parking]])

    scaled_data = scaler.transform(data)

    prediction = model.predict(scaled_data)

    st.success(f"Predicted Price: ₹ {prediction[0]:,.2f}")