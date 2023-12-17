from dotenv import load_dotenv
import os

load_dotenv() 


import pyaudio
import webrtcvad
import numpy as np
import noisereduce as nr
import soundfile as sf
import openai
import time

def record_and_transcribe(start_time):
    starting_time = time.time() - start_time
    print(f"Input module starting time: {starting_time} seconds")

    # Initialize OpenAI client
    openai.api_key = os.getenv('OPENAI.API_KEY')

    # Function to save audio
    def save_audio(data, rate, filename='processed_output.wav'):
        sf.write(filename, data, rate)

    # Initialize PyAudio and VAD
   
    vad = webrtcvad.Vad(3)  # Higher aggressiveness
    audio = pyaudio.PyAudio()


    FORMAT = pyaudio.paInt16
    CHANNELS = 1
    RATE = 48000
    CHUNK = int(RATE * 0.02)  # Frame size

    stream = audio.open(format=FORMAT, channels=CHANNELS, rate=RATE, input=True, frames_per_buffer=CHUNK)
   

    is_recording = False
    no_speech_count = 0
    recorded_frames = []

    SPEECH_THRESHOLD = 5
    NO_SPEECH_LIMIT = int(2 * RATE / CHUNK)  # 2 seconds of silence
    MIN_RECORDING_DURATION = 1  # seconds

    init_time = time.time() - start_time - starting_time
    print(f"Input module initialization time: {init_time} seconds")

    try:
        while True:
            frame = stream.read(CHUNK, exception_on_overflow=False)
            is_speech = vad.is_speech(frame, RATE)

            if is_speech:
                no_speech_count = 0
                if not is_recording:
                    is_recording = True
                    recorded_frames = []
                recorded_frames.append(np.frombuffer(frame, dtype=np.int16))
            elif is_recording:
                no_speech_count += 1
                if no_speech_count >= NO_SPEECH_LIMIT:
                    is_recording = False
                    audio_data = np.concatenate(recorded_frames)

                    # Calculate the duration in seconds
                    recording_duration = len(audio_data) / RATE

                    if recording_duration >= MIN_RECORDING_DURATION:
                        
                        speech_time = time.time() - start_time - init_time
                        print(f"speech detected: {speech_time} seconds")
                        # Perform noise reduction
                        reduced_noise_audio = nr.reduce_noise(y=audio_data, sr=RATE)

                        # Save the processed audio
                        save_audio(reduced_noise_audio, RATE)

                        before_time = time.time() - start_time - speech_time
                        print(f"before whisper API time: {before_time} seconds")

                        # Send file to Whisper for transcription
                        transcription = openai.audio.transcriptions.create(model="whisper-1", file=open('processed_output.wav', 'rb'))
                        print("Transcription:", transcription.text)
                        
                        after_time = time.time() - start_time - before_time
                        print(f"After whisper API time: {after_time} seconds")
                        return transcription.text  # Return the transcribed text
                    
                    else:
                        print(f"Recording too short ({recording_duration:.2f}s), discarding.")
                    recorded_frames = []

    except KeyboardInterrupt:
        pass
    finally:
        stream.stop_stream()
        stream.close()
        audio.terminate() 

if __name__ == "__main__":
    print(record_and_transcribe())