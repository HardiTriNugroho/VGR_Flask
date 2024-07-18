from flask import Flask, request, jsonify, render_template
import numpy as np
import tensorflow as tf
import librosa
import os
from PreProccesing import preprocess_audio  # Ensure this module exists

app = Flask(__name__)

# Load your model
model = tf.keras.models.load_model('model/VGR_LIB_augmented_v2_EP30.h5')

def extract_mfcc(audio, sample_rate=16000, n_mfcc=13):
    mfcc = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=n_mfcc)
    return mfcc

def pad_or_truncate(mfcc, max_length=300):
    if mfcc.shape[1] > max_length:
        return mfcc[:, :max_length]
    else:
        pad_width = max_length - mfcc.shape[1]
        return np.pad(mfcc, ((0, 0), (0, pad_width)), mode='constant')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400

    file_path = 'temp.wav'
    file.save(file_path)

    audio, sample_rate = preprocess_audio(file_path)
    if audio is None:
        return jsonify({'error': 'Error in preprocessing audio file'}), 500

    mfcc = extract_mfcc(audio, sample_rate)
    padded_mfcc = pad_or_truncate(mfcc, max_length=300)
    processed_audio = np.expand_dims(padded_mfcc, axis=0)

    prediction = model.predict(processed_audio)
    gender = 'Male' if prediction[0][0] > 0.5 else 'Female'
    accuracy = float(prediction[0][0]) if gender == 'Male' else 1 - float(prediction[0][0])

    return jsonify({'gender': gender, 'accuracy': accuracy})

if __name__ == '__main__':
    app.run(debug=True)
