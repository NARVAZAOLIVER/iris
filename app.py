# app.py

import streamlit as st
import numpy as np
from sklearn.datasets import load_iris
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Title
st.title("🌸 Iris Flower Predictor (Linear Regression)")
st.write("Enter flower features below and predict the species.")

# Load Iris dataset
iris = load_iris()
X = iris.data  # Features
y = iris.target  # Labels
target_names = iris.target_names

# Preprocessing: scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split and train model
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
model = LinearRegression()
model.fit(X_train, y_train)

# Input form
with st.form("iris_form"):
    sepal_length = st.text_input("Sepal Length (cm)", value="5.1")
    sepal_width = st.text_input("Sepal Width (cm)", value="3.5")
    petal_length = st.text_input("Petal Length (cm)", value="1.4")
    petal_width = st.text_input("Petal Width (cm)", value="0.2")
    submit = st.form_submit_button("Predict")

# Predict and display result
if submit:
    try:
        # Convert inputs to float
        input_data = np.array([
            float(sepal_length),
            float(sepal_width),
            float(petal_length),
            float(petal_width)
        ]).reshape(1, -1)

        # Scale input like training data
        input_scaled = scaler.transform(input_data)

        # Predict using the trained model
        prediction = model.predict(input_scaled)
        predicted_class = round(prediction[0])  # Linear Regression gives float
        predicted_class = np.clip(predicted_class, 0, 2)  # Ensure it's within class range

        # Show result
        st.success(f"Predicted Species: **{target_names[predicted_class]}**")

    except ValueError:
        st.error("⚠️ Please enter valid numeric values for all input fields.")
