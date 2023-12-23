import speech_recognition as sr
import sys

# Initialize recognizer
r = sr.Recognizer()

# Function to transcribe speech and log the output
def listen_and_transcribe():
    while True:
        with sr.Microphone() as source:
            # Adjust for ambient noise and set a shorter listening duration
            r.adjust_for_ambient_noise(source, duration=1)

            try:
                print("Listening...")  # Indicate that the system is ready to listen
                audio = r.listen(source, phrase_time_limit=10)
                text = r.recognize_google(audio)
                with open('output_log.log', 'a') as log_file:
                    log_file.write("You said: " + text + "\n")
                return text
            except sr.UnknownValueError:
                with open('output_log.log', 'a') as log_file:
                    log_file.write("Google Speech Recognition could not understand audio\n")
            except sr.RequestError as e:
                with open('output_log.log', 'a') as log_file:
                    log_file.write(f"Could not request results from Google Speech Recognition service; {e}\n")
            except KeyboardInterrupt:
                # Allow the user to exit the loop with a keyboard interrupt (Ctrl+C)
                print("Exiting transcription loop.")
                break
