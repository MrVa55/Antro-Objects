# server_processor.py
import numpy as np
import soundfile as sf
import librosa
import io
from whisper import load_model

model = load_model("base")  # Make sure to choose the correct model size

class ServerProcessor:
    def __init__(self, connection, min_chunk_size, sampling_rate):
        self.connection = connection
        self.min_chunk = min_chunk_size
        self.sampling_rate = sampling_rate

    def receive_audio_chunk(self):
        # Assuming a method to receive audio efficiently
        audio_data = self.connection.receive_audio()
        return np.frombuffer(audio_data, dtype=np.int16)

    def process_audio(self, audio_chunk):
        # Save audio data to a temporary WAV file for Whisper processing
        with sf.SoundFile("temp.wav", mode='w', samplerate=self.sampling_rate, channels=1, format='WAV', subtype='PCM_16') as file:
            file.write(audio_chunk)
        result = model.transcribe("temp.wav")
        return result["text"]

    def process(self):
        # Process the received audio and return the transcription
        audio_chunk = self.receive_audio_chunk()
        if audio_chunk.size > 0:
            transcription = self.process_audio(audio_chunk)
            return transcription
        else:
            print("No audio received or processing complete.")
            return None