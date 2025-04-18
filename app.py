import os
import torch
import librosa
import numpy as np
import joblib
import streamlit as st
from transformers import WhisperProcessor, WhisperForConditionalGeneration
from huggingface_hub import hf_hub_download
import torch.nn as nn
import tempfile

# Check for available device
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
LABELS = ["HC", "MCI", "Dementia"]
REPO_ID = "gandalf513/memotagdementia"

# Define model architecture
class SimpleAudioTextClassifier(nn.Module):
    def __init__(self, text_feat_dim, acoustic_feat_dim=16, num_labels=3):
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Linear(text_feat_dim + acoustic_feat_dim, 64),
            nn.ReLU(),
            nn.Linear(64, num_labels)
        )

    def forward(self, text_feats, acoustic_feats):
        combined = torch.cat((text_feats, acoustic_feats), dim=1)
        return self.classifier(combined)

# Function to extract acoustic features
def extract_acoustic_features(wav_path):
    y, sr = librosa.load(wav_path, sr=16000)
    duration = librosa.get_duration(y=y, sr=sr)
    tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
    pitch = np.mean(librosa.yin(y, fmin=50, fmax=300))
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    mfcc_mean = np.mean(mfcc, axis=1)
    features = [duration, tempo, pitch] + mfcc_mean.tolist()
    return torch.tensor(features, dtype=torch.float32).unsqueeze(0).to(DEVICE)

# Function to load models
@st.cache_resource
def load_models():
    # Load Whisper
    whisper_processor = WhisperProcessor.from_pretrained("openai/whisper-tiny")
    whisper_model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-tiny").to(DEVICE)
    whisper_model.eval()
    
    # Load TF-IDF vectorizer
    vectorizer_path = hf_hub_download(repo_id=REPO_ID, filename="tfidf_vectorizer.pkl")
    tfidf_vectorizer = joblib.load(vectorizer_path)
    
    # Load model
    model = SimpleAudioTextClassifier(text_feat_dim=256).to(DEVICE)
    model_path = hf_hub_download(repo_id=REPO_ID, filename="pytorch_model.bin")
    model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    model.eval()
    
    return whisper_processor, whisper_model, tfidf_vectorizer, model

# Function to process audio and make prediction
def process_audio(audio_file, whisper_processor, whisper_model, tfidf_vectorizer, model):
    # Save audio to temp file
    with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp_file:
        tmp_file.write(audio_file.getbuffer())
        audio_path = tmp_file.name
    
    try:
        # Transcribe audio
        audio, sr = librosa.load(audio_path, sr=16000)
        input_features = whisper_processor(audio, sampling_rate=sr, return_tensors="pt").input_features.to(DEVICE)
        
        with torch.no_grad():
            # Generate transcription
            predicted_ids = whisper_model.generate(input_features)
            transcription = whisper_processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]
            
            # TF-IDF transform
            text_feats = tfidf_vectorizer.transform([transcription]).toarray()
            text_feats_tensor = torch.tensor(text_feats, dtype=torch.float32).to(DEVICE)
            
            # Extract acoustic features
            acoustic_feats = extract_acoustic_features(audio_path)
            
            # Predict
            outputs = model(text_feats_tensor, acoustic_feats)
            predicted_class = torch.argmax(outputs, dim=1).item()
            prediction = LABELS[predicted_class]
            
            return transcription, prediction
    
    finally:
        # Clean up temp file
        if os.path.exists(audio_path):
            os.remove(audio_path)
        # Free up memory
        torch.cuda.empty_cache() if torch.cuda.is_available() else None

# Streamlit UI
def main():
    st.set_page_config(page_title="Dementia Audio Analysis", page_icon="🎙️")
    
    st.title("Dementia Audio Analysis")
    st.write("Upload an audio recording to analyze for cognitive health indicators.")
    
    # Model loading with spinner
    with st.spinner("Loading models... This may take a moment."):
        whisper_processor, whisper_model, tfidf_vectorizer, model = load_models()
    
    # File upload
    audio_file = st.file_uploader("Upload Audio File", type=["wav", "mp3", "ogg"])
    
    if audio_file is not None:
        st.audio(audio_file)
        
        if st.button("Analyze Audio"):
            with st.spinner("Analyzing audio... Please wait."):
                try:
                    transcription, prediction = process_audio(
                        audio_file, whisper_processor, whisper_model, tfidf_vectorizer, model
                    )
                    
                    # Display results
                    st.success("Analysis complete!")
                    st.subheader("Results")
                    
                    st.markdown("**Transcription:**")
                    st.write(transcription)
                    
                    st.markdown("**Prediction:**")
                    if prediction == "HC":
                        label = "Healthy Control"
                    elif prediction == "MCI":
                        label = "Mild Cognitive Impairment"
                    else:
                        label = "Dementia"
                    
                    st.write(f"{prediction} ({label})")
                    
                    # Add disclaimer
                    st.caption("Note: This is an automated screening tool and not a medical diagnosis. Please consult with healthcare professionals for proper evaluation.")
                    
                except Exception as e:
                    st.error(f"An error occurred during analysis: {str(e)}")
    
    # Add information section
    with st.expander("About this tool"):
        st.write("""
        This tool uses machine learning to analyze speech patterns and acoustic features to screen for signs 
        of cognitive impairment. The analysis is based on both what is said (transcription) and how it is said 
        (acoustic features like rhythm, pitch, and pauses).
        
        The model classifies recordings into three categories:
        - **HC**: Healthy Control - No signs of cognitive impairment
        - **MCI**: Mild Cognitive Impairment - Subtle signs of cognitive changes
        - **Dementia**: Pattern consistent with dementia
        
        This tool is for screening purposes only and should not replace professional medical advice.
        """)

if __name__ == "__main__":
    main()
