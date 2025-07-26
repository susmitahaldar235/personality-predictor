import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Load your saved model
model = joblib.load("my_project.pkl")  # your trained SVM model   

# Title
st.title("🧠 Personality Type Predictor")
st.write("Predict whether someone is an **Introvert** or **Extrovert** based on their social behavior.")

# Input form
time_spent_alone = st.slider("Time Spent Alone (hours/day)", 0.0, 24.0, 0.0)
stage_fear = st.selectbox("Do you have stage fear?", ["Yes", "No"])
social_event_attendance = st.slider("Social Event Attendance (per week)", 0.0, 10.0, 0.0)
going_outside = st.slider("How often do you go outside? (0-10)", 0, 10, 0)
drained_after_socializing = st.selectbox("Do you feel drained after socializing?", ["Yes", "No"])
friends_circle_size = st.slider("Friends Circle Size", 0.0, 15.0, 0.0)
post_frequency = st.slider("Social Media Post Frequency (per week)", 0.0, 15.0, 0.0)

# 🔁 Convert categorical inputs from Yes/No to 1/0
stage_fear_val = 1 if stage_fear == "Yes" else 0
drained_val = 1 if drained_after_socializing == "Yes" else 0

# Combine into a feature array
features = np.array([[time_spent_alone, stage_fear_val, social_event_attendance,
                      going_outside, drained_val, friends_circle_size, post_frequency]])

# Prediction
if st.button("Predict"):
    prediction = model.predict(features)[0]
    label = "Introvert" if prediction == 1 else "Extrovert"
    st.success(f"Predicted Personality: **{label}**")

