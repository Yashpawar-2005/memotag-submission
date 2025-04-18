# import os
# import torch
# import librosa
# import numpy as np
# from flask import Flask, request, jsonify, render_template_string
# from transformers import RobertaTokenizer, RobertaModel, WhisperProcessor, WhisperForConditionalGeneration
# from huggingface_hub import hf_hub_download
# import torch.nn as nn

# app = Flask(__name__)
# DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# LABELS = ["HC", "MCI", "Dementia"]

# # ------------------ Load Models ------------------
# MODEL_REPO = "gandalf513/memotagdementia"

# # Load tokenizer
# tokenizer = RobertaTokenizer.from_pretrained(MODEL_REPO)

# # Load Whisper processor + model
# whisper_processor = WhisperProcessor.from_pretrained("openai/whisper-base")
# whisper_model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-base").to(DEVICE)
# whisper_model.eval()

# # Define model architecture
# class RobertaAudioClassifier(nn.Module):
#     def __init__(self, num_labels=3, acoustic_feat_dim=16):
#         super(RobertaAudioClassifier, self).__init__()
#         self.roberta = RobertaModel.from_pretrained("roberta-base")
#         self.classifier = nn.Sequential(
#             nn.Linear(768 + acoustic_feat_dim, 64),
#             nn.ReLU(),
#             nn.Linear(64, num_labels)
#         )

#     def forward(self, input_ids, attention_mask, acoustic_feats):
#         roberta_output = self.roberta(input_ids=input_ids, attention_mask=attention_mask)
#         cls_output = roberta_output.last_hidden_state[:, 0, :]
#         combined = torch.cat((cls_output, acoustic_feats), dim=1)
#         return self.classifier(combined)

# # Download and load model weights
# model = RobertaAudioClassifier().to(DEVICE)
# model_path = hf_hub_download(repo_id=MODEL_REPO, filename="pytorch_model.bin")
# model.load_state_dict(torch.load(model_path, map_location=DEVICE))
# model.eval()

# # ------------------ Feature Extraction ------------------
# def extract_acoustic_features(wav_path):
#     y, sr = librosa.load(wav_path, sr=16000)
#     duration = librosa.get_duration(y=y, sr=sr)
#     tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
#     pitch = np.mean(librosa.yin(y, fmin=50, fmax=300))
#     mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
#     mfcc_mean = np.mean(mfcc, axis=1)
#     features = [duration, tempo, pitch] + mfcc_mean.tolist()
#     return torch.tensor(features, dtype=torch.float32).unsqueeze(0).to(DEVICE)

# # HTML template for the upload page
# HTML_TEMPLATE = '''
# <!DOCTYPE html>
# <html lang="en">
# <head>
#     <meta charset="UTF-8">
#     <meta name="viewport" content="width=device-width, initial-scale=1.0">
#     <title>Dementia Audio Analysis</title>
#     <style>
#         body {
#             font-family: Arial, sans-serif;
#             max-width: 800px;
#             margin: 0 auto;
#             padding: 20px;
#         }
#         h1 {
#             color: #2c3e50;
#             text-align: center;
#         }
#         .container {
#             border: 1px solid #ddd;
#             padding: 20px;
#             border-radius: 5px;
#             margin-top: 20px;
#         }
#         .form-group {
#             margin-bottom: 15px;
#         }
#         label {
#             display: block;
#             margin-bottom: 5px;
#             font-weight: bold;
#         }
#         #result {
#             margin-top: 20px;
#             padding: 15px;
#             border: 1px solid #ddd;
#             border-radius: 5px;
#             display: none;
#         }
#         button {
#             background-color: #3498db;
#             color: white;
#             border: none;
#             padding: 10px 15px;
#             border-radius: 5px;
#             cursor: pointer;
#         }
#         button:hover {
#             background-color: #2980b9;
#         }
#     </style>
# </head>
# <body>
#     <h1>Dementia Audio Analysis</h1>
#     <div class="container">
#         <form id="uploadForm" enctype="multipart/form-data">
#             <div class="form-group">
#                 <label for="audioFile">Upload Audio File:</label>
#                 <input type="file" id="audioFile" name="file" accept="audio/*" required>
#             </div>
#             <button type="submit">Analyze</button>
#         </form>
#     </div>
#     <div id="loading" style="display: none; text-align: center; margin-top: 20px;">
#     <p>Analyzing...</p>
# </div>
#     <div id="result">
#         <h2>Analysis Results</h2>
#         <p><strong>Transcription:</strong> <span id="transcription"></span></p>
#         <p><strong>Prediction:</strong> <span id="prediction"></span></p>
#     </div>

#     <script>
#         document.getElementById('uploadForm').addEventListener('submit', function(e) {
#             e.preventDefault();
#              document.getElementById('loading').style.display = 'block';
#     // Hide any previous results
#     document.getElementById('result').style.display = 'none';
#             const formData = new FormData();
#             const fileInput = document.getElementById('audioFile');
            
#             formData.append('file', fileInput.files[0]);
            
#             fetch('/predict', {
#                 method: 'POST',
#                 body: formData
#             })
#             .then(response => response.json())
#             .then(data => {
#              document.getElementById('loading').style.display = 'none';
#                 document.getElementById('transcription').textContent = data.transcription;
#                 document.getElementById('prediction').textContent = data.prediction;
#                 document.getElementById('result').style.display = 'block';
#             })
#             .catch(error => {
#                 console.error('Error:', error);
#                 alert('An error occurred during analysis. Please try again.');
#             });
#         });
#     </script>
# </body>
# </html>
# '''

# # ------------------ Flask Routes ------------------
# @app.route("/", methods=["GET"])
# def index():
#     return render_template_string(HTML_TEMPLATE)

# @app.route("/predict", methods=["POST"])
# def predict():
#     if 'file' not in request.files:
#         return jsonify({"error": "No file uploaded"}), 400

#     file = request.files['file']
#     file_path = f"./temp_audio.{file.filename.split('.')[-1]}"
#     file.save(file_path)

#     try:
#         # Transcribe
#         audio, sr = librosa.load(file_path, sr=16000)
#         input_features = whisper_processor(audio, sampling_rate=sr, return_tensors="pt").input_features.to(DEVICE)
#         predicted_ids = whisper_model.generate(input_features)
#         transcription = whisper_processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]

#         # Encode text
#         encoded = tokenizer(transcription, return_tensors="pt", padding="max_length", truncation=True, max_length=512)
#         input_ids = encoded['input_ids'].to(DEVICE)
#         attention_mask = encoded['attention_mask'].to(DEVICE)

#         # Acoustic features
#         acoustic_feats = extract_acoustic_features(file_path)

#         # Model inference
#         with torch.no_grad():
#             outputs = model(input_ids=input_ids, attention_mask=attention_mask, acoustic_feats=acoustic_feats)
#             predicted_class = torch.argmax(outputs, dim=1).item()

#         return jsonify({
#             "transcription": transcription,
#             "prediction": LABELS[predicted_class]
#         })

#     except Exception as e:
#         return jsonify({"error": str(e)}), 500
#     finally:
#         if os.path.exists(file_path):
#             os.remove(file_path)

# # ------------------ Run ------------------
# if __name__ == "__main__":
#     app.run(debug=True)

import os
import torch
import librosa
import numpy as np
import joblib
from flask import Flask, request, jsonify, render_template_string
from transformers import WhisperProcessor, WhisperForConditionalGeneration
from huggingface_hub import hf_hub_download
import torch.nn as nn

app = Flask(__name__)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
LABELS = ["HC", "MCI", "Dementia"]
REPO_ID = "gandalf513/memotagdementia"

# Load Whisper
whisper_processor = WhisperProcessor.from_pretrained("openai/whisper-tiny")
whisper_model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-tiny").to(DEVICE)
whisper_model.eval()

# Load TF-IDF vectorizer
vectorizer_path = hf_hub_download(repo_id=REPO_ID, filename="tfidf_vectorizer.pkl")
tfidf_vectorizer = joblib.load(vectorizer_path)

# Define model
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

# Load model
model = SimpleAudioTextClassifier(text_feat_dim=256).to(DEVICE)
model_path = hf_hub_download(repo_id=REPO_ID, filename="pytorch_model.bin")
model.load_state_dict(torch.load(model_path, map_location=DEVICE))
model.eval()

# Extract acoustic features
def extract_acoustic_features(wav_path):
    y, sr = librosa.load(wav_path, sr=16000)
    duration = librosa.get_duration(y=y, sr=sr)
    tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
    pitch = np.mean(librosa.yin(y, fmin=50, fmax=300))
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    mfcc_mean = np.mean(mfcc, axis=1)
    features = [duration, tempo, pitch] + mfcc_mean.tolist()
    return torch.tensor(features, dtype=torch.float32).unsqueeze(0).to(DEVICE)

# Web UI HTML
# Same as your original HTML (omitted for brevity)
# HTML template for the upload page
HTML_TEMPLATE = '''
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Dementia Audio Analysis</title>
    <style>
        body {
            font-family: Arial, sans-serif;
            max-width: 800px;
            margin: 0 auto;
            padding: 20px;
        }
        h1 {
            color: #2c3e50;
            text-align: center;
        }
        .container {
            border: 1px solid #ddd;
            padding: 20px;
            border-radius: 5px;
            margin-top: 20px;
        }
        .form-group {
            margin-bottom: 15px;
        }
        label {
            display: block;
            margin-bottom: 5px;
            font-weight: bold;
        }
        #result {
            margin-top: 20px;
            padding: 15px;
            border: 1px solid #ddd;
            border-radius: 5px;
            display: none;
        }
        button {
            background-color: #3498db;
            color: white;
            border: none;
            padding: 10px 15px;
            border-radius: 5px;
            cursor: pointer;
        }
        button:hover {
            background-color: #2980b9;
        }
    </style>
</head>
<body>
    <h1>Dementia Audio Analysis</h1>
    <div class="container">
        <form id="uploadForm" enctype="multipart/form-data">
            <div class="form-group">
                <label for="audioFile">Upload Audio File:</label>
                <input type="file" id="audioFile" name="file" accept="audio/*" required>
            </div>
            <button type="submit">Analyze</button>
        </form>
    </div>
    <div id="loading" style="display: none; text-align: center; margin-top: 20px;">
    <p>Analyzing...</p>
</div>
    <div id="result">
        <h2>Analysis Results</h2>
        <p><strong>Transcription:</strong> <span id="transcription"></span></p>
        <p><strong>Prediction:</strong> <span id="prediction"></span></p>
    </div>

    <script>
        document.getElementById('uploadForm').addEventListener('submit', function(e) {
            e.preventDefault();
             document.getElementById('loading').style.display = 'block';
    // Hide any previous results
    document.getElementById('result').style.display = 'none';
            const formData = new FormData();
            const fileInput = document.getElementById('audioFile');
            
            formData.append('file', fileInput.files[0]);
            
            fetch('/predict', {
                method: 'POST',
                body: formData
            })
            .then(response => response.json())
            .then(data => {
             document.getElementById('loading').style.display = 'none';
                document.getElementById('transcription').textContent = data.transcription;
                document.getElementById('prediction').textContent = data.prediction;
                document.getElementById('result').style.display = 'block';
            })
            .catch(error => {
                console.error('Error:', error);
                alert('An error occurred during analysis. Please try again.');
            });
        });
    </script>
</body>
</html>
'''

@app.route("/", methods=["GET"])
def index():
    return render_template_string(HTML_TEMPLATE)

@app.route("/predict", methods=["POST"])
def predict():
    if 'file' not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files['file']
    file_path = f"./temp_audio.{file.filename.split('.')[-1]}"
    file.save(file_path)

    try:
        # Transcribe audio
        audio, sr = librosa.load(file_path, sr=16000)
        input_features = whisper_processor(audio, sampling_rate=sr, return_tensors="pt").input_features.to(DEVICE)
        predicted_ids = whisper_model.generate(input_features)
        transcription = whisper_processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]

        # TF-IDF transform
        text_feats = tfidf_vectorizer.transform([transcription]).toarray()
        text_feats_tensor = torch.tensor(text_feats, dtype=torch.float32).to(DEVICE)

        # Acoustic
        acoustic_feats = extract_acoustic_features(file_path)

        # Predict
        with torch.no_grad():
            outputs = model(text_feats_tensor, acoustic_feats)
            predicted_class = torch.argmax(outputs, dim=1).item()

        return jsonify({
            "transcription": transcription,
            "prediction": LABELS[predicted_class]
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500
    finally:
        if os.path.exists(file_path):
            os.remove(file_path)

if __name__ == "__main__":
    app.run(debug=True)
