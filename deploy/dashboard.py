import streamlit as st
import pandas as pd
import joblib

# --- 1. Load your trained model ---
# This file must be in the same GitHub repo
MODEL_FILE = 'cpu_model_pipeline.pkl'

@st.cache_resource
def load_model(model_path):
    """Loads the model from a .pkl file"""
    try:
        model = joblib.load(model_path)
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

model = load_model(MODEL_FILE)

# --- 2. Build the Dashboard UI ---
st.set_page_config(layout="centered")
st.title("CPU Usage Predictor 🤖")
st.write("A free dashboard hosted on Streamlit Community Cloud.")

# [cite_start]--- Create input forms for all features [cite: 306-311] ---
st.header("Input Features")
col1, col2 = st.columns(2)

with col1:
    cpu_request = st.number_input("CPU Request (cores)", min_value=0.1, value=1.0, step=0.1)
    mem_request = st.number_input("Memory Request (MiB)", min_value=64, value=256, step=64)
    cpu_limit = st.number_input("CPU Limit (cores)", min_value=0.1, value=2.0, step=0.1)
    
with col2:
    mem_limit = st.number_input("Memory Limit (MiB)", min_value=128, value=512, step=64)
    runtime_minutes = st.number_input("Runtime (Minutes)", min_value=1, value=60)
    controller_kind = st.selectbox("Controller Kind", 
                                   options=["ReplicaSet", "Deployment", "StatefulSet", "Job"])

# --- 3. Prediction Logic ---
if st.button("Predict CPU Usage", type="primary") and model is not None:

    # 1. Format the data into a DataFrame (Scikit-learn expects it)
    input_data = {
        "cpu_request": [cpu_request],
        "mem_request": [mem_request],
        "cpu_limit": [cpu_limit],
        "mem_limit": [mem_limit],
        "runtime_minutes": [runtime_minutes],
        "controller_kind": [controller_kind]
    }
    
    input_df = pd.DataFrame(input_data)
    
    # 2. Make prediction
    try:
        st.info("Making prediction...")
        prediction = model.predict(input_df)
        
        # Display the result
        st.metric(label="Predicted CPU Usage (cores)", value=f"{prediction[0]:.4f}")
        st.success("Prediction successful!")

    except Exception as e:
        st.error(f"An error occurred during prediction: {e}")