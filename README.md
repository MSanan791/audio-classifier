🎧 Deep Learning Audio Classifier

An end-to-end Machine Learning pipeline for audio processing and classification. This repository contains all the necessary scripts to process .wav files, extract audio features, train a deep neural network, visualize training performance, and test the model using both pre-recorded files and live microphone input.

🎯 Project Overview: What Are We Trying to Achieve?

The primary goal of this project is to build a robust Audio Classification System. Whether it is meant for recognizing speech commands, identifying environmental sounds, or classifying speaker identities, the core objective remains the same: teaching a machine to "listen" and categorize audio signals accurately.

Raw audio data is highly complex and continuous, making it difficult for standard neural networks to process directly. Therefore, this project implements a specialized pipeline to bridge the gap between raw soundwaves and machine-understandable predictions.

🧠 In-Depth Analysis: How Are We Achieving It?

To achieve high-accuracy audio classification, the project is broken down into four distinct phases:

1. Data Preprocessing & Feature Extraction (X_features.npy, Y_labels.npy)

Raw .wav files contain thousands of amplitude samples per second. Instead of feeding raw audio to the model, we extract meaningful acoustic features—most likely MFCCs (Mel-Frequency Cepstral Coefficients) or Mel Spectrograms.

These features represent the power spectrum of a sound, effectively mimicking how the human ear perceives frequencies.

By saving these extracted features and their corresponding labels as NumPy arrays (X_features.npy and Y_labels.npy), we drastically reduce loading times and computation overhead during the training phase.

2. Model Architecture & Training (main.py, my_model.keras)

The core intelligence of the system lies in the neural network trained in main.py.

The Model: The network (likely a Convolutional Neural Network (CNN) or a deep Dense network) takes the 2D feature arrays as input. It learns the temporal and frequency-based patterns associated with different audio classes.

Training: The model iterates through the data, adjusting its weights using an optimizer (like Adam) and a loss function (like Categorical Crossentropy).

Artifacts: The successfully trained model is serialized and saved as my_model.keras and my_model.h5 for future use, preventing the need to retrain from scratch.

3. Performance Visualization (Results.py, training_history.json)

Deep learning requires careful monitoring to prevent issues like overfitting. During training, metrics (accuracy, loss, validation accuracy, and validation loss) are dumped into training_history.json. The Results.py script parses this data to generate visualizations, allowing developers to analyze how well the model generalized to unseen data over time.

4. Real-World Inference (test.py, RecordAudio.py)

A model is only as good as its real-world application.

test.py: Runs inference on isolated sample files (like sample_audio.wav or AAMNG1232.wav) to verify accuracy.

RecordAudio.py: Takes the project a step further by capturing live audio from the user's microphone, instantly extracting its features, passing it through my_model.keras, and predicting the result in real-time.

📂 Repository Structure

File

Description

main.py

The core training script. Loads features, defines the model, trains it, and saves the output.

test.py

Inference script to test the saved model against specific .wav files.

RecordAudio.py

Script to record live audio from your microphone and classify it in real-time.

Results.py

Generates graphs and charts using the data from training_history.json.

suleman.py

Auxiliary script (likely used for custom data preprocessing, augmentation, or alternative modeling).

X_features.npy

Pre-extracted numerical features from the audio dataset.

Y_labels.npy

Corresponding categorical labels for the X_features.npy data.

my_model.keras / .h5

The trained and saved neural network models.

training_history.json

A JSON log of the model's loss and accuracy at each training epoch.

*.wav

Sample audio files used for testing inference capabilities.

⚙️ Complete Setup & Installation Guide

Follow these steps to set up the environment and run the project on your local machine.

Step 1: Prerequisites

Ensure you have Python 3.8 to 3.10 installed on your system. You will also need pip for installing dependencies.
(Note: Python 3.11+ sometimes faces compatibility issues with certain older audio libraries, so 3.9/3.10 is recommended).

Step 2: Clone the Repository

Open your terminal or command prompt and clone the repository:

git clone <your-repository-url>
cd audio-classifier


Step 3: Create a Virtual Environment (Recommended)

Creating a virtual environment ensures that the project dependencies do not interfere with your global Python packages.

# Windows
python -m venv venv
venv\Scripts\activate

# macOS/Linux
python3 -m venv venv
source venv/bin/activate


Step 4: Install Dependencies

While a requirements.txt is not explicitly listed, the files imply the use of several standard machine learning and audio processing libraries. Install them using the following command:

pip install numpy pandas librosa tensorflow keras matplotlib sounddevice soundfile scipy


Note: sounddevice and soundfile are required for RecordAudio.py to capture live microphone input.

Step 5: Verify the Pre-processed Data

Ensure that X_features.npy and Y_labels.npy are in the root directory. If you are starting fresh with new audio, you will need to run your feature extraction script (potentially located within main.py or suleman.py) to generate these files from your .wav dataset.

🚀 Usage Guide

1. Training the Model

If you want to train the model from scratch using the provided .npy files:

python main.py


This will output a new my_model.keras file and overwrite the training_history.json.

2. Visualizing Results

To see how well the model trained (Accuracy vs. Epochs, Loss vs. Epochs):

python Results.py


This will open up matplotlib graphs showing the training curves.

3. Testing on Existing Audio

To run a prediction on one of the sample files (e.g., sample_audio.wav):

python test.py


(Make sure to edit test.py if you need to point it to a specific .wav file path).

4. Live Audio Classification

To test the model with your own voice or live sounds:

python RecordAudio.py


The script will prompt you to speak/make a sound for a few seconds, process the recording, and output the model's classification.

🛠️ Troubleshooting

PortAudio Error when running RecordAudio.py: If you are on Mac/Linux, sounddevice requires the PortAudio library.

Mac: brew install portaudio

Ubuntu/Debian: sudo apt-get install libportaudio2

TensorFlow GPU Issues: If you are trying to train on a GPU, ensure you have the correct CUDA and cuDNN versions installed corresponding to your TensorFlow version.

Developed with ❤️ using Python and Deep Learning.
