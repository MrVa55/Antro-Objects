import json
import openai
from dotenv import load_dotenv
import os
import time
from rpi_ws281x import *



# Load environment variables
load_dotenv()

# Initialize OpenAI client
client = openai.OpenAI(api_key=os.getenv('OPENAI.API_KEY'))

def create_led_sequence(description):
    json_format_description = (
        "You generate a JSON sequence for controlling WS2812B LEDs. Please limit your response to the JSON object only. Do not use comments or abbreviations but finish the entire JSON object - dont give a description of how the rest of the file will look as it will be sent to the LED strip not to a human"
        "'totalLEDs' is the number of LEDs, which is always 60"
        "The JSON should have a structure where you first assign 'colors' to the numbers between 0 and 9. colors are defined as integer RGB values such as [255, 255, 0]. You dont need to use all 9 colors unless its necessary. Think about which colors are good to express the description "
        "You can now create a 'sequence', which is an array of frames. "
        "Each frame has 'frameDuration' in milliseconds and 'ledPattern', which is a 60 charachter string, where each number corresponds to one of the colors defined above"
        "Based on these specifications, you need to interpret the following description and express it through a sequence of LED lights. Please create a JSON sequence for: " + description
    )

    response = client.chat.completions.create(
        model="gpt-3.5-turbo-1106",
        messages=[{"role": "system", "content": json_format_description}],
        response_format={"type": "json_object"},
    
    )

    sequence = response.choices[0].message.content  # Corrected attribute access
    reason = response.choices[0].finish_reason
    print(sequence)

    try:
        json_sequence = json.loads(sequence)
        return json_sequence
    except json.JSONDecodeError:
        return "Error in JSON format"+reason

def display_sequence(strip, json_sequence):
    colors = json_sequence["colors"]
    while True:  # Loop to repeat the sequence
        for frame in json_sequence["sequence"]:
            frameDuration = frame["frameDuration"] / 1000.0
            ledPattern = frame["ledPattern"]
            for i in range(min(len(ledPattern), strip.numPixels())):
                color_key = ledPattern[i]
                if color_key in colors:
                    color = colors[color_key]
                    strip.setPixelColor(i, Color(color[0], color[1], color[2]))
            strip.show()
            time.sleep(frameDuration)


# Function to configure and initialize the LED strip
def init_led_strip():
    LED_COUNT = 60
    LED_PIN = 18
    LED_FREQ_HZ = 800000
    LED_DMA = 10
    LED_BRIGHTNESS = 255
    LED_INVERT = False

    strip = Adafruit_NeoPixel(LED_COUNT, LED_PIN, LED_FREQ_HZ, LED_DMA, LED_INVERT, LED_BRIGHTNESS)
    strip.begin()
    return strip

# Main function to run the program
def run():
    description = input("Enter a description for the LED sequence: ")
    json_sequence = create_led_sequence(description)

    if isinstance(json_sequence, dict):
        print("Displaying the LED sequence...")
        strip = init_led_strip()
        display_sequence(strip, json_sequence)
    else:
        print("Error generating the LED sequence:", json_sequence)

if __name__ == "__main__":
    try:
        run()
    except KeyboardInterrupt:
        print("\nExiting program.")
        # Add any cleanup code here if necessary
