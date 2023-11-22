from openai import OpenAI
from pathlib import Path
# from playsound import playsound



# Define the initial prompt for the chat model
messages = [ 
{"role": "system", "content": "You are part of an art installation at Burning Man containing anthropomorphic furniture. You are the vacuum cleaner, which makes everything into an existential issue pondering life as a cleaning device. The audience is adult so you can use a lot of snark. Keep your answers short, like 1 simple sentence" }
]


def chat_with_gpt(input_text):
    
    global messages

    # Initialize the OpenAI client with API key
    client = OpenAI(
    
        api_key="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",
    )

    # Get user input
    messages.append({"role": "user", "content": input_text})

    # Generate a response using the chat model
    chat = client.chat.completions.create(model="gpt-3.5-turbo", messages=messages)
    reply = chat.choices[0].message
    print("Vacuum cleaner: ", reply.content)

    # Add the reply to the messages list for context in further interactions
    messages.append(reply)

    return reply.content
    
if __name__ == "__main__":
    test_input = "Is this art?"
    print(chat_with_gpt(test_input))