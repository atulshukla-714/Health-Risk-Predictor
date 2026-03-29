import streamlit as st
import pickle
import numpy as np

model = pickle.load(open("model/model.pkl", "rb"))

st.title("Health Risk Predictor")

age = st.slider("Age", 10, 80)
bmi = st.slider("BMI", 10, 40)
smoking = st.selectbox("Smoking", ["No", "Yes"])
alcohol = st.selectbox("Alcohol", ["Low", "Medium", "High"])
exercise = st.slider("Exercise (days/week)", 0, 7)
bp = st.slider("Blood Pressure", 100, 200)
sugar = st.slider("Sugar Level", 70, 200)

smoking_val = 1 if smoking == "Yes" else 0
alcohol_map = {"Low": 0, "Medium": 1, "High": 2}
alcohol_val = alcohol_map[alcohol]

if st.button("Predict"):
    input_data = np.array([[age, bmi, smoking_val, alcohol_val, exercise, bp, sugar]])
    prediction = model.predict(input_data)

    risk_map = {0: "Low", 1: "Medium", 2: "High"}
    st.success(f"Predicted Health Risk: {risk_map[prediction[0]]}")
