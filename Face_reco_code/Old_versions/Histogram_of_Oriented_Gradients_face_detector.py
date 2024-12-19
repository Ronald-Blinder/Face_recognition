import dlib
import cv2
import time
import threading

# Initialize the webcam
cap = cv2.VideoCapture(0)

# Check if camera is opened successfully
if not cap.isOpened():
    print("Error: Could not open video stream.")
    exit()

# Set the frame rate (optional)
cap.set(cv2.CAP_PROP_FPS, 30)

# Use HOG face detector for faster performance
hog_face_detector = dlib.get_frontal_face_detector()

# To track performance
frame_count = 0
start_time_total = time.time()

# Lock for thread synchronization
frame_lock = threading.Lock()
frame_buffer = None

# Function to capture frames in a separate thread
def capture_frames():
    global frame_buffer
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame")
            break
        # Resize the frame to speed up processing (optional)
        frame_resized = cv2.resize(frame, (640, 480))  # Resize to a smaller size
        with frame_lock:
            frame_buffer = frame_resized

# Start the frame capture thread
capture_thread = threading.Thread(target=capture_frames)
capture_thread.daemon = True
capture_thread.start()

while True:
    frame_count += 1
    start_time = time.time()  # Start timer for frame processing

    # Wait for the next frame
    with frame_lock:
        if frame_buffer is None:
            continue
        frame_resized = frame_buffer

    # Detect faces in the frame using HOG detector
    faces = hog_face_detector(frame_resized)

    # Draw rectangles around detected faces
    for face in faces:
        # Get the coordinates of the bounding box
        x1, y1, x2, y2 = (face.left(), face.top(), face.right(), face.bottom())

        # Draw a rectangle around the face
        cv2.rectangle(frame_resized, (x1, y1), (x2, y2), (0, 255, 0), 2)

    # Display the resulting frame with face detection
    cv2.imshow("Webcam", frame_resized)

    # Calculate frame processing time and display it
    frame_processing_time = time.time() - start_time
    print(f"Frame {frame_count}: {frame_processing_time:.3f} seconds")

    # Check if the user pressed the 'q' key to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Calculate and print total time for all frames processed
end_time_total = time.time()
total_processing_time = end_time_total - start_time_total
print(f"Total processing time for {frame_count} frames: {total_processing_time:.3f} seconds")

# Release the capture and close any open windows
cap.release()
cv2.destroyAllWindows()
