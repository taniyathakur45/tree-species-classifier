import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
import json
import warnings
warnings.filterwarnings('ignore')

# Load the model
try:
    model = tf.keras.models.load_model('tree_species_model.h5')
    st.success("✅ Model loaded successfully!")
except Exception as e:
    st.error(f"❌ Error loading model: {e}")
    st.stop()

# Load species mapping
try:
    with open('species_mapping.json', 'r') as f:
        species_mapping = json.load(f)
    st.success("✅ Species mapping loaded!")
except Exception as e:
    st.warning("⚠️ Species mapping not found. Will show IDs only.")
    species_mapping = None

# Title
st.title("🌳 Tree Intelligence Assistant")

# Sidebar
st.sidebar.header("Enter Tree Data")

# Input fields - only the 4 features the model expects
latitude = st.sidebar.number_input("Latitude", value=35.0, format="%.6f")
longitude = st.sidebar.number_input("Longitude", value=-106.0, format="%.6f")
diameter = st.sidebar.number_input("Diameter (cm)", value=20.0, min_value=0.0)
height = st.sidebar.number_input("Height (m)", value=10.0, min_value=0.0)

# Display what the model expects
st.sidebar.subheader("Model Information")
st.sidebar.info(f"Model expects: 4 numerical features")
st.sidebar.info(f"Output classes: 96 tree species")

# Convert inputs to the exact format the model expects (4 features)
input_data = np.array([[latitude, longitude, diameter, height]])

# Display input data for verification
st.sidebar.subheader("Input Data")
input_df = pd.DataFrame({
    'Feature': ['Latitude', 'Longitude', 'Diameter (cm)', 'Height (m)'],
    'Value': [latitude, longitude, diameter, height]
})
st.sidebar.dataframe(input_df)

if st.button("🔍 Predict Tree Species"):
    try:
        # Make prediction with the correct input format
        prediction = model.predict(input_data)
        predicted_class = np.argmax(prediction)
        confidence = np.max(prediction) * 100
        
        # Get species name if mapping is available
        if species_mapping and str(predicted_class) in species_mapping:
            species_info = species_mapping[str(predicted_class)]
            species_name = species_info['name']
            scientific_name = species_info['scientific_name']
            st.success(f"🌱 Predicted Tree Species: **{species_name}** ({scientific_name})")
        else:
            st.success(f"🌱 Predicted Tree Species ID: {predicted_class}")
        
        st.info(f"Confidence: {confidence:.2f}%")
        
        # Display prediction probabilities
        st.subheader("Top 10 Predictions")
        prob_data = []
        for i in range(len(prediction[0])):
            prob = prediction[0][i] * 100
            if species_mapping and str(i) in species_mapping:
                species_name = species_mapping[str(i)]['name']
                scientific_name = species_mapping[str(i)]['scientific_name']
                prob_data.append({
                    'Species ID': i,
                    'Common Name': species_name,
                    'Scientific Name': scientific_name,
                    'Probability (%)': prob
                })
            else:
                prob_data.append({
                    'Species ID': i,
                    'Common Name': f'Species {i}',
                    'Scientific Name': 'Unknown',
                    'Probability (%)': prob
                })
        
        prob_df = pd.DataFrame(prob_data).sort_values('Probability (%)', ascending=False)
        st.dataframe(prob_df.head(10))
        
        # Show the input that was used
        st.subheader("Input Used for Prediction")
        st.code(f"Latitude: {latitude}, Longitude: {longitude}, Diameter: {diameter}cm, Height: {height}m")
        
    except Exception as e:
        st.error(f"❌ Prediction error: {e}")
        st.info("💡 This might be due to input data format mismatch with the model's expected features.")
