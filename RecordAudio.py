import sounddevice as sd
import wave


def record_audio(file_name="recorded_audio.wav", duration=5, sample_rate=44100):
    """
    Records audio from the microphone and saves it as a WAV file.

    Args:
    - file_name (str): Name of the output WAV file.
    - duration (int): Duration of the recording in seconds.
    - sample_rate (int): Sample rate for the audio recording.
    """
    print(f"Recording for {duration} seconds...")
    # Record audio
    audio_data = sd.rec(int(duration * sample_rate), samplerate=sample_rate, channels=1, dtype='int16')
    sd.wait()  # Wait for the recording to finish
    print("Recording complete!")

    # Save as a WAV file
    with wave.open(file_name, "w") as wf:
        wf.setnchannels(1)  # Mono channel
        wf.setsampwidth(2)  # 2 bytes per sample (16-bit PCM)
        wf.setframerate(sample_rate)
        wf.writeframes(audio_data.tobytes())

    print(f"Audio saved as {file_name}")


# Record a 5-second audio and save it in the current directory
record_audio(file_name="sample_audio.wav", duration=5)
