import input_module
import chat_module
import playback_module
import time

def main():


    while True:
        start_time = time.time()
        
        # Record and transcribe user input
        user_input = input_module.record_and_transcribe(start_time)
        input_time = time.time() - start_time
        print(f"Input module processing time: {input_time} seconds")

        # Get response from ChatGPT
        gpt_response = chat_module.chat_with_gpt(user_input)
        chat_time = time.time() - start_time - input_time
        print(f"Chat module processing time: {chat_time} seconds")

        # Read aloud the response
        playback_module.text_to_speech_and_play(gpt_response)
        playback_time = time.time() - start_time - chat_time
        print(f"Playback module processing time: {playback_time} seconds")

if __name__ == "__main__":
    main()