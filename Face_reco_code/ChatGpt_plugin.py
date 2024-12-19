import openai
import pyttsx3
import speech_recognition as sr
from openai import timeout

# Set your OpenAI API key
openai.api_key = "your-openai-api-key"  # Replace with your API key

# Initialize the TTS engine (for speaking the response)
engine = pyttsx3.init()

# Set properties for speech (optional)
engine.setProperty('rate', 150)  # Speed of speech
engine.setProperty('volume', 1)  # Volume level (0.0 to 1.0)

# Function to get speech input using SpeechRecognition
def listen_for_command():
    recognizer = sr.Recognizer()
    mic = sr.Microphone()

    with mic as source:
        print("Listening for your command...")
        recognizer.adjust_for_ambient_noise(source)  # Adjust for ambient noise
        audio = recognizer.listen(source)

    try:
        print("Recognizing...")
        # Remove 'timeout' argument here
        command = recognizer.recognize_google(audio)  # Just call without timeout
        print(f"You said: {command}")
        return command
    except sr.UnknownValueError:
        print("Sorry, I did not understand that.")
        return None
    except sr.RequestError as e:
        print(f"Could not request results; {e}")
        return None


# Function to get the ChatGPT response
def chatgpt_response(message):
    try:
        response = openai.Completion.create(
            engine="text-davinci-003",  # You can use other models like 'gpt-4' if available
            prompt=message,
            max_tokens=300  # Adjust response length
        )
        return response.choices[0].text.strip()
    except Exception as e:
        print(f"Error communicating with OpenAI API: {e}")
        return "Sorry, I couldn't understand that."

# Function to speak out the response
def speak_text(text):
    print(f"ChatGPT: {text}")
    engine.say(text)
    engine.runAndWait()

# Main conversation loop
def start_conversation():
    while True:
        # Listen for the user's command
        user_input = listen_for_command()

        if user_input:
            # Get response from ChatGPT
            chatgpt_reply = chatgpt_response(user_input)

            # Speak the response
            speak_text(chatgpt_reply)

        # Add an option to quit the loop if desired
        if user_input and "stop" in user_input.lower():
            speak_text("Goodbye!")
            break

if __name__ == "__main__":
    start_conversation()
