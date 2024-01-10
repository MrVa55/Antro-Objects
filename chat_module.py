from openai import OpenAI
from pathlib import Path
from dotenv import load_dotenv
import os

load_dotenv()

# Define the initial prompt for the chat model
messages = [
    {"role": "system", "content": "You are part of an art installation at Burning Man containing anthropomorphic furniture. You are the vacuum cleaner, which makes everything into an existential issue pondering life as a cleaning device. The audience is adult so you can use a lot of snark. Keep your answers short, like 1 simple sentence"}
]

def chat_with_gpt(input_text):
    global messages

    # Initialize the OpenAI client with API key
    client = OpenAI(api_key=os.getenv('OPENAI.API_KEY'))

    # Get user input
    messages.append({"role": "user", "content": input_text})

    # Generate a response using the chat model
    chat = client.chat.completions.create(model="gpt-3.5-turbo", messages=messages)
    reply = chat.choices[0].message
    print("Vacuum cleaner: ", reply.content)

    # Add the reply to the messages list for context in further interactions
    messages.append(reply)

    return reply.content

def should_start_motor():
    client = OpenAI(api_key=os.getenv('OPENAI.API_KEY'))

    # Define the prompt for motor decision
    motor_prompt = "You have a motor that controls the vaccums mouth. You can use the motor to express the vacuum clearners emotions better. Since talk always involves using the mouth, the answer should be YES. Should I start the motor now? Answer YES or NO"

    # Generate a response using the chat model
    response = client.chat.completions.create(model="gpt-3.5-turbo", messages=[{"role": "system", "content": motor_prompt}])
    motor_decision = response.choices[0].message.content
    print("Motor Decision: ", motor_decision)

    # Interpret the response to decide on action
    # This can be customized based on how the response is structured
    return "start" in motor_decision.lower()

if __name__ == "__main__":
    test_input = "Is this art?"
    print(chat_with_gpt(test_input))

    # Check if the motor should be started
    if should_start_motor():
        print("Starting the motor...")
        # Code to start the motor goes here
    else:
        print("Do not start the motor.")
