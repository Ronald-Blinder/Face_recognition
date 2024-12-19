import os
import torch
from facenet_pytorch import MTCNN, InceptionResnetV1
import cv2
import time
import numpy as np
from torchvision.models import resnet18, ResNet18_Weights
import torch.nn.functional as F
from torchvision import transforms

# Check if CUDA is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Initialize MTCNN for face detection
mtcnn = MTCNN(keep_all=True, device=device)

# Initialize InceptionResnetV1 model for face recognition
face_recognition_model = InceptionResnetV1(pretrained='vggface2').eval().to(device)

# Initialize ResNet18 model (if you need it for some other purpose)
weights = ResNet18_Weights.IMAGENET1K_V1  # or ResNet18_Weights.DEFAULT for the most up-to-date weights
resnet_model = resnet18(weights=weights)
resnet_model = resnet_model.to(device)

#Initialize webcam - choose 0 for web cam or 1 for Canon camera
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
    # Normalize to the same range used by InceptionResnetV1
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

# Load reference images from the folder
folder_path = r"C:\Users\g_epgsv_lab\Downloads\Pictures_for_recognition"   # Change to your folder path
reference_embeddings = load_reference_images_from_folder(folder_path)

if not reference_embeddings:
    print("No reference images found in the folder.")
    exit()

# Initialize frame count and start time
frame_count = 0
start_time = time.time()

# Get the processing interval from the user
# process_interval = int(input("Enter the frame interval to process (e.g., 1 for every frame, 2 to skip one frame, etc.): "))
process_interval=3
# Set the desired width and height for the display window
display_width = 800     # or any value you want
display_height = int(display_width/1.33)  # or any value you want

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame")
        break

    frame_count += 1  # Increment frame counter

    # Skip frames not divisible by process_interval
    if frame_count % process_interval != 0:
        continue

    # Resize the frame to the desired size
    frame_resized = cv2.resize(frame, (display_width, display_height))

    # Convert the resized frame to RGB
    rgb_frame = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)

    # Detect faces
    start_detection = time.time()
    boxes, _ = mtcnn.detect(rgb_frame)
    detection_time = time.time() - start_detection

    # Draw boxes around faces and perform face recognition
    if boxes is not None:
        for box in boxes:
            x1, y1, x2, y2 = map(int, box)
            cv2.rectangle(frame_resized, (x1, y1), (x2, y2), (0, 255, 0), 2)

            # Extract face and compute its embedding
            face = rgb_frame[y1:y2, x1:x2]
            face = preprocess(face)  # Preprocess the face
            face_embedding = face_recognition_model(face.unsqueeze(0).to(device))

            # Compare embeddings with reference embeddings to see if it's a match
            match_found = False
            for person_name, ref_embedding in reference_embeddings.items():
                similarity = F.cosine_similarity(ref_embedding, face_embedding)
                if similarity > 0.65:  # Threshold can be adjusted based on the model's performance
                    label = f"{person_name}"  # First line: person's name
                    match_label = f"Match {similarity.item() * 100:.2f}%"  # Second line: match percentage
                    match_found = True
                    break

            if not match_found:
                label = f"Random"  # Display Random if no match is found

            # Display the name (on the first line)
            cv2.putText(frame_resized, label, (x1 , y2 + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)

            # Display the match percentage (on the second line)
            if match_found:
                cv2.putText(frame_resized, match_label, (x1 , y2 + 35), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)

    # Display the resized frame
    cv2.imshow("Face Recognition", frame_resized)

    # Print frame processing time
    print(f"Frame {frame_count} processed in {detection_time:.3f} seconds")

    # Break on pressing 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Calculate total time and FPS
end_time = time.time()
total_time = end_time - start_time
fps = frame_count / total_time
print(f"Processed {frame_count} frames in {total_time:.2f} seconds ({fps:.2f} FPS)")

# Release resources
cap.release()
cv2.destroyAllWindows()
