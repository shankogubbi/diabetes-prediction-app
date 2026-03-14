import streamlit as st
import numpy as np
import joblib

model = joblib.load('model.pkl')
scaler = joblib.load('scaler.pkl')

st.title("Diabetes Prediction App")
st.write("Enter patient details below to predict diabetes risk.")

col1, col2 = st.columns(2)

with col1:
    pregnancies = st.number_input("Pregnancies", 0, 20, 1)
    glucose     = st.number_input("Glucose", 0, 300, 120)
    bp          = st.number_input("Blood Pressure", 0, 150, 70)
    skin        = st.number_input("Skin Thickness", 0, 100, 20)

with col2:
    insulin = st.number_input("Insulin", 0, 900, 80)
    bmi     = st.number_input("BMI", 0.0, 70.0, 25.0)
    dpf     = st.number_input("Diabetes Pedigree Function", 0.0, 3.0, 0.5)
    age     = st.number_input("Age", 1, 120, 25)

if st.button("Predict"):
    sample = np.array([[pregnancies, glucose, bp, skin,
                        insulin, bmi, dpf, age]])
    sample_scaled = scaler.transform(sample)
    prediction = model.predict(sample_scaled)[0]
    prob = model.predict_proba(sample_scaled)[0][1]

    if prediction == 1:
        st.error(f"Result: Diabetic — {prob*100:.1f}% confidence")
    else:
        st.success(f"Result: Not Diabetic — {(1-prob)*100:.1f}% confidence")
