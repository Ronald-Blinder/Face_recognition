import os
import cv2
import time

# Initialize the webcam
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Unable to access the webcam.")
    exit()

# Ask for the person's name
person_name = input("Enter the person's name: ")

# Create a folder for the person if it doesn't exist
person_folder = os.path.join("C:\\Users\\g_epgsv_lab\\Downloads", person_name)
if not os.path.exists(person_folder):
    os.makedirs(person_folder)

print(f"Capturing images for {person_name}. Press 'Q' to stop capturing.")

# Initialize frame count for naming images
image_count = 0

# Capture images from the webcam
while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame")
        break

    # Display the webcam feed
    cv2.imshow(f"Capturing {person_name}", frame)

    # Wait for keypress
    key = cv2.waitKey(1) & 0xFF

    if key == ord('q'):  # Press 'Q' to stop capturing
        print("Capture stopped.")
        break

    # Save the captured image every 1 second (or adjust the interval as needed)
    image_count += 1
    image_path = os.path.join(person_folder, f"{person_name}_{image_count}.jpg")
    cv2.imwrite(image_path, frame)
    print(f"Image {image_count} saved to {image_path}")

    # Wait for 1 second before taking the next photo
    time.sleep(1)

# Release the webcam and close any OpenCV windows
cap.release()
cv2.destroyAllWindows()

print(f"Images for {person_name} have been saved in {person_folder}.")
