import os
import librosa
import soundfile as sf
import numpy as np
from scipy.signal import butter, lfilter
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def load_audio(file_path, sr=22050):
    try:
        audio, sample_rate = librosa.load(file_path, sr=sr)
        return audio, sample_rate
    except Exception as e:
        logging.error(f"Error loading {file_path}: {e}")
        return None, None

def high_pass_filter(audio, sample_rate, cutoff=100):
    nyquist = 0.5 * sample_rate
    norm_cutoff = cutoff / nyquist
    b, a = butter(1, norm_cutoff, btype='high', analog=False)
    filtered_audio = lfilter(b, a, audio)
    return filtered_audio

def normalize_audio(audio):
    norm_audio = audio / np.max(np.abs(audio))
    return norm_audio

def resample_audio(audio, original_sr, target_sr=16000):
    resampled_audio = librosa.resample(audio, orig_sr=original_sr, target_sr=target_sr)
    return resampled_audio, target_sr

def remove_silence(audio, top_db=20):
    non_silent_intervals = librosa.effects.split(audio, top_db=top_db)
    non_silent_audio = np.concatenate([audio[start:end] for start, end in non_silent_intervals])
    return non_silent_audio

def preprocess_audio(file_path, target_sr=16000, high_pass_cutoff=100, top_db=20):
    try:
        audio, sample_rate = load_audio(file_path)
        if audio is None:
            return None, None
        
        audio = high_pass_filter(audio, sample_rate, cutoff=high_pass_cutoff)
        audio = normalize_audio(audio)
        audio, sample_rate = resample_audio(audio, sample_rate, target_sr=target_sr)
        audio = remove_silence(audio, top_db=top_db)
        
        return audio, sample_rate
    except Exception as e:
        logging.error(f"Error processing {file_path}: {e}")
        return None, None
