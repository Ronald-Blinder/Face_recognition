import torch
from facenet_pytorch import MTCNN
import cv2
import time
from torchvision.models import resnet18, ResNet18_Weights

# Check if CUDA is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Initialize MTCNN for face detection
mtcnn = MTCNN(keep_all=True, device=device)

# Initialize ResNet18 model with weights (updating to the latest method)
weights = ResNet18_Weights.IMAGENET1K_V1  # or ResNet18_Weights.DEFAULT for the most up-to-date weights
resnet_model = resnet18(weights=weights)
resnet_model = resnet_model.to(device)  # Move the model to the appropriate device

# Initialize webcam
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Unable to access the webcam.")
    exit()

frame_count = 0
start_time = time.time()

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame")
        break

    # Convert the frame to RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Detect faces
    start_detection = time.time()
    boxes, _ = mtcnn.detect(rgb_frame)
    detection_time = time.time() - start_detection

    # Draw boxes around faces
    if boxes is not None:
        for box in boxes:
            x1, y1, x2, y2 = map(int, box)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

    # Display the frame
    cv2.imshow("Face Detection", frame)
    frame_count += 1

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
