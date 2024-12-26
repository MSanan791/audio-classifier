import os
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"  # Disable OneDNN optimizations for compatibility
import keras.src.saving.saving_api
# TensorFlow/Keras Imports
import tensorflow as tf
from tensorflow.keras.regularizers import l2
import json
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
from tensorflow.keras.models import load_model

import json


X = np.load("X_features.npy")
Y = np.load("Y_labels.npy")

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, random_state=1)
# Load the saved model
model = load_model("my_model.keras")

# Use the model for predictions or further training


# Load the history from a JSON file
with open("training_history.json", "r") as json_file:
    history = json.load(json_file)

model.summary()

# Evaluate the model
predictions = model.predict(X_test)
score = model.evaluate(X_test, Y_test)
print("Test Score:", score)
keras.src.saving.saving_api.save_model(model,"my_model.keras")  # Save the model as an keras file

y_pred = np.argmax(predictions, axis=1)  # Predictions from the model
y_true = np.argmax(Y_test, axis=1)

report = classification_report(y_true, y_pred, target_names=['air_conditioner', 'car_horn',
'children_playing', 'dog_bark', 'drilling', 'engine_idling', 'gun_shot', 'jackhammer', 'siren','street_music'])
print("Classification Report:")
print(report)


plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.legend()
plt.show()

# Plot training and validation loss
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.legend()
plt.show()