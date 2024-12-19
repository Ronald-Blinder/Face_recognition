import os
import cv2
import pyttsx3
import torch
from facenet_pytorch import MTCNN, InceptionResnetV1
import torch.nn.functional as F
from torchvision import transforms
import time

# Initialize MTCNN for face detection
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
mtcnn = MTCNN(keep_all=True, device=device)

# Initialize InceptionResnetV1 model for face recognition
face_recognition_model = InceptionResnetV1(pretrained='vggface2').eval().to(device)

# Initialize text-to-speech engine
engine = pyttsx3.init()

# set voices
voices = engine.getProperty('voices')
desired_voice_index = 1  # Choose the voice index you prefer
engine.setProperty('voice', voices[desired_voice_index].id)

# Set slower speech rate
rate = engine.getProperty('rate')  # Get the current speech rate
engine.setProperty('rate', rate - 50)  # Decrease the rate (default is around 200)

# Initialize webcam
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Unable to access the webcam.")
    exit()

# Define preprocessing transformations
preprocess = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((160, 160)),  # Resize to 160x160
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

# Load reference images from subfolders
def load_reference_images_from_folder(folder_path):
    reference_embeddings = {}
    for subfolder_name in os.listdir(folder_path):
        subfolder_path = os.path.join(folder_path, subfolder_name)
        if os.path.isdir(subfolder_path):
            image_files = [f for f in os.listdir(subfolder_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
            if image_files:
                reference_image_path = os.path.join(subfolder_path, image_files[0])
                reference_image = cv2.imread(reference_image_path)
                reference_image_rgb = cv2.cvtColor(reference_image, cv2.COLOR_BGR2RGB)
                boxes, _ = mtcnn.detect(reference_image_rgb)
                if boxes is not None:
                    for box in boxes:
                        x1, y1, x2, y2 = map(int, box)
                        face = reference_image_rgb[y1:y2, x1:x2]
                        face = preprocess(face)  # Preprocess the face
                        face_embedding = face_recognition_model(face.unsqueeze(0).to(device))
                        reference_embeddings[subfolder_name] = face_embedding
    return reference_embeddings

# Folder containing reference images for recognized faces
folder_path = r"C:\Users\g_epgsv_lab\Downloads\Pictures_for_recognition"
reference_embeddings = load_reference_images_from_folder(folder_path)

if not reference_embeddings:
    print("No reference images found in the folder.")
    exit()

# Dictionary to store the last time each person was greeted
last_greeted = {}

# Function to announce a name using text-to-speech
def announce_name(name):
    # Announce the person's name via text-to-speech
    engine.say(f"Welcome {name}")
    engine.runAndWait()

# Start processing frames from the webcam
while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame")
        break

    # Convert the frame to RGB for face detection
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Detect faces
    boxes, _ = mtcnn.detect(rgb_frame)

    if boxes is not None:
        for box in boxes:
            x1, y1, x2, y2 = map(int, box)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

            # Extract face and compute its embedding
            face = rgb_frame[y1:y2, x1:x2]
            face = preprocess(face)  # Preprocess the face
            face_embedding = face_recognition_model(face.unsqueeze(0).to(device))

            # Compare embeddings with reference embeddings
            match_found = False
            for person_name, ref_embedding in reference_embeddings.items():
                similarity = F.cosine_similarity(ref_embedding, face_embedding)
                if similarity > 0.65:  # Threshold for matching faces
                    label = f"Hello, {person_name}!"

                    # Check if the person was greeted recently (within the last 10 seconds)
                    current_time = time.time()
                    if person_name not in last_greeted or (current_time - last_greeted[person_name]) > 15:
                        # Announce the person's name loudly
                        announce_name(person_name)

                        # Update the last greeted time for this person
                        last_greeted[person_name] = current_time
                    match_found = True
                    break

            if not match_found:
                label = "Unknown Person"

            # Display the name (on the first line)
            cv2.putText(frame, label, (x1, y2 + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)

    # Display the frame with the face detection box and name
    cv2.imshow("Face Recognition", frame)

    # Break on pressing 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
