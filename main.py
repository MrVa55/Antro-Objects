# main.py
from server_processor import ServerProcessor
from connection import Connection  # Assuming this handles the socket connection
import chat_module
import playback_module
import socket

def main():
    # Set up the server socket
    host, port = 'localhost', 43007
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind((host, port))
        s.listen(1)
        print('Server listening on', host, port)

        # Accept a connection
        conn, addr = s.accept()
        with conn:
            print('Connected by', addr)
            connection = Connection(conn)
            processor = ServerProcessor(connection, min_chunk_size=1024, sampling_rate=16000)

            # Process audio and interact via chat
            transcription = processor.process()
            if transcription:
                print("Transcription:", transcription)
                response = chat_module.chat_with_gpt(transcription)
                print("Chat Response:", response)
                playback_module.text_to_speech_and_play(response)

if __name__ == "__main__":
    main()
