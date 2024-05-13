from openai import OpenAI
from pathlib import Path
from dotenv import load_dotenv
import os

import json
import random

load_dotenv()

# Load characters from a JSON file
def load_characters(directory):
    characters = []
    directory_path = Path(directory)
    for file in directory_path.glob('*.json'):
        with open(file, 'r') as f:
            characters.append(json.load(f))
    return characters



# Define the initial prompt for the chat model with all characters
def initialize_messages(characters):
    character_descriptions = []
    for character in characters:
        description = f"Charachter name:{character['character']} Characther instructions: {character['instructions']}"
        character_descriptions.append(description)
    
    initial_prompt = ("You manage an art installation at Burning Man containing anthropomorphic furniture. "
                      "Each time you receive a reply, you must choose a character from the list below and answer as that characther. "
                      "Preface your answer with the name of the object." 
                      "It is important, that your responses are brief, succinct  and natural like a line in a theater manuscript. It should be speech-like and not written language.  Ideally, your replies should be one sentence, to keep the conversation flowing smoothly and engagingly."
                      "The audience is mature, so you can use humor, swear words and snark. When choosing which characther to reply as, decide which one will be able to give them most surprising and funny answer. The list of characthers are:" + "\n".join(character_descriptions))
    return [{"role": "system", "content": initial_prompt}]


def chat_with_gpt(input_text):
    global messages
    # Initialize the OpenAI client with API key
    client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))

    # Get user input
    messages.append({"role": "user", "content": input_text})

    # Generate a response using the chat model
    chat = client.chat.completions.create(model="gpt-3.5-turbo", messages=messages)
    reply = chat.choices[0].message

    # Add the reply to the messages list for context in further interactions
    messages.append(reply)

    # Parse the character name
    if ':' in reply.content:
        character_name, response = reply.content.split(':', 1)
        character_name = character_name.strip()
        response = response.strip()
    else:
        character_name = "Unknown"
        response = reply.content

    # Return both the character name and the response
    return character_name, response


if __name__ == "__main__":

    # Load characters from JSON and initialize messages
    characters = load_characters('object-characters/')
    messages = initialize_messages(characters)

    # Simulate a conversation starting with a user input
    test_input = "Is this art?"
    character, response = chat_with_gpt(test_input)
    print(f"{character}: {response}")

  