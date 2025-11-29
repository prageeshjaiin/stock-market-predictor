import streamlit as st
import numpy as np
import pickle

# ---------------- PAGE CONFIG ----------------
st.set_page_config(page_title="Tesla Stock Price Predictor", layout="wide")

# ---------------- LOAD MODEL ----------------
model = pickle.load(open("model.pkl", "rb"))

# ---------------- APP TITLE ----------------
st.title("🚗 Tesla Stock Price Predictor")
st.write("Predict Tesla's next **closing price** using your trained Machine Learning model.")

# ---------------- USER INPUT SECTION ----------------
st.subheader("🧮 Enter Tesla Stock Values")

open_val = st.number_input("Open Price", value=200.0)
high_val = st.number_input("High Price", value=205.0)
low_val = st.number_input("Low Price", value=198.0)
vol_val = st.number_input("Volume", value=1000000.0)

# ---------------- PREDICT BUTTON ----------------
if st.button("🔮 Predict Closing Price"):
    X_input = np.array([[open_val, high_val, low_val, vol_val]])
    predicted_price = model.predict(X_input)[0]
    
    st.success(f"Predicted Closing Price: **${predicted_price:.2f}**")
