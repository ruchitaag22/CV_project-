import cv2
import sys

def detect_faces_haarcascade_video(cascade_path):
    """
    Detects faces in a live video feed using OpenCV's Haar Cascade classifier.

    Args:
        cascade_path (str): The path to the Haar Cascade XML file for face detection.
                            Defaults to 'haarcascade_frontalface_default.xml',
                            assuming it's in the same directory.
    """
    # Load the pre-trained Haar Cascade classifier for face detection
    face_cascade = cv2.CascadeClassifier(cascade_path)

    # Check if the cascade classifier loaded successfully
    if face_cascade.empty():
        print(f"Error: Could not load face cascade XML file from {cascade_path}")
        print("Please ensure 'haarcascade_frontalface_default.xml' is in the correct directory.")
        sys.exit(1) # Exit if cascade file is not found

    # Open the default webcam (usually index 0)
    # You can change the index if you have multiple cameras (e.g., 1, 2, etc.)
    cap = cv2.VideoCapture(0)

    # Check if the webcam was opened successfully
    if not cap.isOpened():
        print("Error: Could not open video stream. Make sure a webcam is connected and accessible.")
        sys.exit(1) # Exit if webcam cannot be accessed

    print("Press 'q' to quit the video feed.")

    while True:
        # Read a frame from the video feed
        ret, frame = cap.read()

        # If frame is not read correctly, break the loop
        if not ret:
            print("Failed to grab frame, exiting...")
            break

        # Convert the frame to grayscale, as Haar Cascades work best on grayscale images
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Perform face detection on the grayscale frame
        # Parameters are similar to image detection, but might need fine-tuning for video
        faces = face_cascade.detectMultiScale(
            gray_frame,
            scaleFactor=1.1,     # How much the image size is reduced at each image scale
            minNeighbors=5,      # How many neighbors each candidate rectangle should have
            minSize=(30, 30),    # Minimum possible object size
            # maxSize=(200, 200) # Optional: uncomment and adjust if you want to limit max face size
        )

        # Draw rectangles around the detected faces
        for (x, y, w, h) in faces:
            # Draw a rectangle (image, top-left corner, bottom-right corner, color, thickness)
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2) # Green rectangle, 2 pixels thick
        flip = cv2.flip(frame,1)
        # Display the frame with detected faces
        cv2.imshow('Live Face Detection', flip)

        # Break the loop if 'q' key is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the video capture object and close all OpenCV windows
    cap.release()
    cv2.destroyAllWindows()

# --- Example Usage ---
if __name__ == "__main__":
    # Make sure 'haarcascade_frontalface_default.xml' is in the same directory
    # or provide its full path.
    cascade_file = 'haarcascade_frontalface_alt.xml'

    detect_faces_haarcascade_video(cascade_file)
