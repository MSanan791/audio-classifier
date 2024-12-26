import os
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"  # Disable OneDNN optimizations for compatibility
import keras.src.saving.saving_api
# TensorFlow/Keras Imports
import tensorflow as tf
from tensorflow.keras.regularizers import l2

from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dropout, Dense
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import LearningRateScheduler
# Data Analysis and Visualization
import pandas as pd
pd.plotting.register_matplotlib_converters()
import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt

# Audio Processing
import librosa
import librosa.display

# Machine Learning Tools
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report
from sklearn.preprocessing import MinMaxScaler

# Other
import glob

# Function to parse the dataset
from scipy.signal import butter, lfilter

# High-pass and low-pass filter design
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

# Function to parse the dataset
def parser(row):
    feature = []
    label = []

    # Define the target size for the Mel spectrogram
    target_size = 128  # Adjust based on your needs

    for i in range(len(row)):  # Process all rows
        file_name = './Data/audio/fold' + str(df['fold'][i]) + '/' + df["slice_file_name"][i]
        print(f"{i} / {len(row)}: {file_name}")

        # Load the audio file
        X, sample_rate = librosa.load(file_name, res_type='kaiser_fast')

        # Apply bandpass filter to remove noise
        filtered_signal = bandpass_filter(X, lowcut=100.0, highcut=8000.0, fs=sample_rate)

        # Normalize the signal
        filtered_signal = librosa.util.normalize(filtered_signal)

        # Compute the Mel spectrogram / mfcc  default value 13
        mel_spectrogram = librosa.feature.melspectrogram(y=filtered_signal, sr=sample_rate)

        # Normalize by averaging across the time axis
        mel_spectrogram = np.mean(mel_spectrogram.T, axis=0)

        # Ensure mel_spectrogram has a consistent size (pad or truncate)
        if len(mel_spectrogram) < target_size:
            mel_spectrogram = np.pad(mel_spectrogram, (0, target_size - len(mel_spectrogram)), mode='constant')
        elif len(mel_spectrogram) > target_size:
            mel_spectrogram = mel_spectrogram[:target_size]

        # Append the Mel spectrogram and label to respective lists
        feature.append(mel_spectrogram)
        label.append(df["classID"][i])  # Assuming classID contains the labels

    # Convert lists into numpy arrays
    feature_array = np.array(feature)
    label_array = np.array(label)

    return feature_array, label_array


# Load the dataset
df = pd.read_csv("./Data/metadata/UrbanSound8K.csv")

num_classes = df["classID"].nunique()
print("Number of classes:", num_classes)

# Parse features and labels
# X, Y = parser(df)

# np.save("X_features.npy", X)  # Save features
# np.save("Y_labels.npy", Y)    # Save labels
#
# print("Features and labels saved successfully!")

X = np.load("X_features.npy")
Y = np.load("Y_labels.npy")

# print("Loaded features and labels successfully!")
# print(X.shape, Y.shape)

# Convert labels to categorical (one-hot encoding) after parsing
#Y = to_categorical(Y)
Y = to_categorical(Y, num_classes=num_classes)
# Print the shapes of X and Y to verify consistency
print(X.shape, Y.shape)

# Split the dataset
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, random_state=1)
print(X_train.shape, Y_train.shape)
print(X_test.shape, Y_test.shape)

# Reshape for CNN input
X_train = X_train.reshape(X_train.shape[0], 16, 8, 1)
X_test = X_test.reshape(X_test.shape[0], 16, 8, 1)

# Define the input dimensions
input_dim = (16, 8, 1)

dat1, sampling_rate1 = librosa.load('./Data/audio/fold5/100032-3-0-0.wav')


plt.figure(figsize=(20, 10))
D = librosa.amplitude_to_db(np.abs(librosa.stft(dat1)), ref=np.max)
plt.subplot(4, 2, 1)
librosa.display.specshow(D, y_axis='linear')
plt.colorbar(format='%+2.0f dB')
plt.title('Linear-frequency power spectrogram')

plt.show()

# Build the model
model = Sequential()
model.add(Conv2D(64, (3, 3), activation='relu', padding='same', input_shape=(16, 8, 1)))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(128, (3, 3), activation='relu', padding='same'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.3))  # Increased dropout rate
model.add(Flatten())
model.add(Dense(256, activation='relu', kernel_regularizer=l2(0.01)))  # Added regularization
model.add(Dense(10, activation='softmax'))


model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

#early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
history = model.fit(X_train, Y_train,
                    validation_data=(X_test, Y_test),
                    epochs=90,
                    batch_size=32,
                    )

# Print the model summary
model.summary()

# Evaluate the model
predictions = model.predict(X_test)
score = model.evaluate(X_test, Y_test)
print("Test Score:", score)
keras.src.saving.saving_api.save_model(model,"my_model.keras")  # Save the model as an keras file

y_pred = np.argmax(predictions, axis=1)  # Predictions from the model
y_true = np.argmax(Y_test, axis=1)

class_mapping = {
    0: "noise",  # air_conditioner
    1: "noise",  # car_horn
    2: "speech",  # children_playing
    3: "speech",  # dog_bark
    4: "noise",  # drilling
    5: "noise",  # engine_idling
    6: "noise",  # gun_shot
    7: "noise",  # jackhammer
    8: "noise",  # siren
    9: "music",  # street_music
}

# Map predictions and ground truth to the new categories
y_true_mapped = [class_mapping[label] for label in y_true]
y_pred_mapped = [class_mapping[label] for label in y_pred]

# Generate the classification report for the mapped classes
unique_classes = ["noise", "speech", "music"]
report_mapped = classification_report(y_true_mapped, y_pred_mapped, target_names=unique_classes)
print("Mapped Classification Report:")
print(report_mapped)
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.legend()
plt.show()

# Plot training and validation loss
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.legend()
plt.show()