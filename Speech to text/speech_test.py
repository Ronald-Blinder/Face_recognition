import speech_recognition as sr

# Initialize recognizer
recognizer = sr.Recognizer()

# Set up the microphone
with sr.Microphone() as source:
    print("Adjusting for ambient noise... Please wait.")
    recognizer.adjust_for_ambient_noise(source, duration=2)  # Adjust for 2 seconds
    print("Listening...")

    # Open a text file to write the recognized text
    with open("speech_to_text.txt", "a") as file:
        while True:
            try:
                # Capture audio from the microphone with no timeout or phrase limit
                audio = recognizer.listen(source)

                # Convert speech to text using the offline engine (pocketsphinx)
                text = recognizer.recognize_sphinx(audio)
                print(f"Recognized Text: {text}")

                # Write the recognized text to the file
                file.write(text + "\n")
                file.flush()  # Ensure text is written immediately

            except sr.UnknownValueError:
                # If speech is not recognized, skip this round
                pass
            except sr.RequestError as e:
                print(f"Could not request results; {e}")
                break
