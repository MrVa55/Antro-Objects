import input_module
import chat_module
import playback_module
import sounddevice
#import time

def main():


    while True:
             
        # Record and transcribe user input
        user_input = input_module.listen_and_transcribe()
        #user_input = "Is this art?"
        print("Transcription:", user_input)

        # Get response from ChatGPT
        gpt_response = chat_module.chat_with_gpt(user_input)
    
        # Read aloud the response
        playback_module.text_to_speech_and_play(gpt_response)

if __name__ == "__main__":
    main()
