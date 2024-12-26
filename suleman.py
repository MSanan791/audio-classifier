import numpy as np
import os
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
import librosa
from keras.models import load_model
from scipy.signal import butter, lfilter

# Load the trained model
model = load_model("my_model.keras")

# High-pass and low-pass filter design for preprocessing
def butter_bandpass(lowcut, highcut, fs, order=5):
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = butter(order, [low, high], btype='band')
    return b, a

def bandpass_filter(data, lowcut, highcut, fs, order=5):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = lfilter(b, a, data)
    return y

# Function to preprocess the audio
def preprocess_audio(file_path, target_size=128):
    # Load the audio file
    X, sample_rate = librosa.load(file_path, res_type='kaiser_fast')

    # Apply bandpass filter to remove noise
    filtered_signal = bandpass_filter(X, lowcut=100.0, highcut=8000.0, fs=sample_rate)

    # Normalize the signal
    filtered_signal = librosa.util.normalize(filtered_signal)

    # Compute the Mel spectrogram
    mel_spectrogram = librosa.feature.melspectrogram(y=filtered_signal, sr=sample_rate)

    # Normalize by averaging across the time axis
    mel_spectrogram = np.mean(mel_spectrogram.T, axis=0)

    # Ensure mel_spectrogram has a consistent size (pad or truncate)
    if len(mel_spectrogram) < target_size:
        mel_spectrogram = np.pad(mel_spectrogram, (0, target_size - len(mel_spectrogram)), mode='constant')
    elif len(mel_spectrogram) > target_size:
        mel_spectrogram = mel_spectrogram[:target_size]

    # Reshape for CNN input
    mel_spectrogram = mel_spectrogram.reshape(1, 16, 8, 1)  # Adjust dimensions for model input
    return mel_spectrogram

X = np.load("X_features.npy")

# Define mapping from original classes to broader categories
class_mapping = {
    0: "noise",  # air_conditioner
    1: "noise",  # car_horn
    2: "speech",  # children_playing
    3: "noise",  # dog_bark
    4: "noise",  # drilling
    5: "noise",  # engine_idling
    6: "music",  # gun_shot
    7: "noise",  # jackhammer
    8: "music",  # siren
    9: "music",  # street_music
}

# Function to classify an audio file
def classify_audio(file_path):
    # Preprocess the audio
    processed_audio = preprocess_audio(file_path)

    # Make a prediction
    prediction = model.predict(processed_audio)
    predicted_class = np.argmax(prediction)

    # Map the predicted class to the broader category
    return class_mapping[predicted_class]


# Test the model on a few sample audio files
sample_files = [
    'AAMNG1232.wav',
    'sample_audio.wav'
]

# Classify and print results
for file in sample_files:
    category = classify_audio(file)
    print(f"File: {file} -> Predicted Category: {category}")
