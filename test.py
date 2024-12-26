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

# Load the features and labels as well as the Model

df = pd.read_csv("./Data/metadata/UrbanSound8K.csv")

num_classes = df["classID"].nunique()

X = np.load("X_features.npy")
Y = np.load("Y_labels.npy")

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

# Compiling the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
history = model.fit(X_train, Y_train,
                    validation_data=(X_test, Y_test),
                    epochs=90,
                    batch_size=32,
                    callbacks=[early_stop])

# Print the model summary
model.summary()

# Evaluate the model
predictions = model.predict(X_test)
score = model.evaluate(X_test, Y_test)
print("Test Score:", score)
keras.src.saving.saving_api.save_model(model,"my_model.keras")  # Save the model as an keras file
# Save the history as a JSON file
with open("training_history.json", "w") as json_file:
    json.dump(history.history, json_file)



