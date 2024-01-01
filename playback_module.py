from openai import OpenAI
import pygame
from pathlib import Path
from dotenv import load_dotenv
import os
import sounddevice

load_dotenv()

def text_to_speech_and_play(text):
    # Initialize the OpenAI client with API key
    openai = OpenAI(
        api_key=os.getenv('OPENAI.API_KEY')
    )
    
    # Specify the path for the output audio file
    speech_file_path = Path(__file__).parent / "speech.mp3"

    # Create a text-to-speech response using the reply content
    response = openai.audio.speech.create(
        model="tts-1",
        voice="onyx",
        input=text  # Use the chat model's reply as the input text
    )

    # Save the audio stream to a file
    response.stream_to_file(speech_file_path)

    # Initialize pygame mixer, Load and play the audio file
    pygame.mixer.init()
    pygame.mixer.music.load(str(speech_file_path))
    pygame.mixer.music.play()

    # Keep the script running until the audio finishes
    while pygame.mixer.music.get_busy():
        pygame.time.Clock().tick(10)

if __name__ == "__main__":
    test_text = "I am doing well, thank you!"
    text_to_speech_and_play(test_text)
