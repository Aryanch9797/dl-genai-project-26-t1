import stremlit as st
import os
import torch
import torch.nn as nn
import librosa
import numpy as np
from src.Deployment_predictions.deployment_inference import load_models, predict_ensemble

st.set_page_config(page_title="Mashup Genre Classifier", layout="centered")

@st.cache_resource
def load_cached_models():
    cnn_path, resnet_path = 'weights/Scratch_CNN.pth', 'weights/ResNet.pth'
    cnn_model, resnet_model = load_models(cnn_path, resnet_path)
    cnn_model.eval()
    resnet_model.eval()
    return cnn_model, resnet_model

with st.spinner("Initializing models into memory..."):
    cnn, resnet = load_cached_models()

st.title("🎵 Noisy Mashup Genre Classifier")
st.markdown("Upload a noisy audio track. The Scratch CNN + ResNet-50 ensemble will extract features and predict the genre.")
    
uploaded_file = st.file_uploader("Upload an audio file (.wav, .mp3)", type=["wav", "mp3"])

if uploaded_file is not None:
    st.audio(uploaded_file, format='audio/wav')
    
    if st.button("Predict Genre"):
        # Create a temporary file path
        temp_audio_path = "temp_uploaded_audio.wav"
        
        # Write the uploaded bytes to disk so librosa can read it
        with open(temp_audio_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
            
        with st.spinner("Analyzing audio..."):
            try:
                # Pass the temporary file and cached models to your inference function
                prediction, confidence = predict_ensemble(
                    audio_path=temp_audio_path, 
                    cnn_model=cnn, 
                    resnet_model=resnet
                )
                st.success("Prediction Complete!")
                st.subheader(f"Predicted Genre: {prediction}")
                st.write(f"Confidence: {confidence:.2%}")
                
            except Exception as e:
                st.error(f"Error during inference: {e}")
                
        # Clean up the file to prevent server storage bloat
        if os.path.exists(temp_audio_path):
            os.remove(temp_audio_path)