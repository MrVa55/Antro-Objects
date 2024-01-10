from openai import OpenAI
import pygame
from pathlib import Path
from dotenv import load_dotenv
import os
import subprocess

load_dotenv()



def text_to_speech_and_play(text):
    openai = OpenAI(api_key=os.getenv('OPENAI.API_KEY'))
    speech_file_path = Path(__file__).parent / "speech.mp3"

    response = openai.audio.speech.create(
        model="tts-1",
        voice="onyx",
        input=text
    )

    response.stream_to_file(speech_file_path)

    pygame.mixer.init()
    pygame.mixer.music.load(str(speech_file_path))
    pygame.mixer.music.play()

    # Start the motor script as a subprocess
    motor_process = subprocess.Popen(['python', 'motormouth.py'])

    while pygame.mixer.music.get_busy():
        pygame.time.Clock().tick(10)

    # Terminate the motor script after audio ends
    motor_process.terminate()

if __name__ == "__main__":
    test_text = "I am doing well, thank you!"
    text_to_speech_and_play(test_text)
