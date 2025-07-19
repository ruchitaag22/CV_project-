import cv2
import sys

def detect_faces_haarcascade_video(cascade_path):  # Fucntion to detect faces in a live video feed using OpenCV's Haar Cascade classifier.
    """
    Args:
        cascade_path (str): The path to the Haar Cascade XML file for face detection.
                            Defaults to 'haarcascade_frontalface_default.xml',
                            assuming it's in the same directory.
    """

    face_cascade = cv2.CascadeClassifier(cascade_path)  # Load the pre-trained Haar Cascade classifier for face detection

    if face_cascade.empty():  # Check if the cascade classifier loaded successfully
        print(f"Error: Could not load face cascade XML file from {cascade_path}")
        print("Please ensure 'haarcascade_frontalface_default.xml' is in the correct directory.")
        sys.exit(1) # Exit if cascade file is not found

    cap = cv2.VideoCapture(0)  # Initialize the default webcam to open it using (0) as the parameter 
    '''# You can change the index if you have multiple cameras (e.g., 1, 2, etc.)'''

    if not cap.isOpened():  # Check if the webcam was opened successfully
        print("Error: Could not open video stream. Make sure a webcam is connected and accessible.")
        sys.exit(1) # Exit if webcam cannot be accessed

    print("Press 'q' to quit the video feed.")

    while True:
        ret, frame = cap.read()  # Read a frame from the video feed and return two values (ret = boolean value indicating teh state of the camera,  frame = image that is being captured by the camera)

        if not ret:  # If frame is not read correctly, break the loop
            print("Failed to grab frame, exiting...")
            break
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # Convert the frame to grayscale because Haar Cascades work best on grayscale images

        # Perform face detection on the grayscale frame
        # Parameters are similar to image detection, but might need fine-tuning for video
        faces = face_cascade.detectMultiScale(
            gray_frame,          # Grayscale photo to process on
            scaleFactor=1.1,     # How much the image size is reduced at each image scale
            minNeighbors=5,      # How many neighbors each candidate rectangle should have
            minSize=(30, 30),    # Minimum possible object size
            # maxSize=(200, 200) # Optional: uncomment and adjust if you want to limit max face size
        )

        # Draw rectangles around the detected faces
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2) # Green rectangle, 2 pixels thick (image, top-left corner, bottom-right corner, color, thickness)

        cv2.imshow('Live Face Detection', frame)  # Display the frame with detected faces
        flipped_frame = cv2.flip(frame,1)
        cv2.imshow('Live Face Detection', flipped_frame) 

        if cv2.waitKey(1) & 0xFF == ord('q'):  # Break the loop if 'q' key is pressed
            break

    cap.release()    # Release the video capture object and close all OpenCV windows
    cv2.destroyAllWindows()


if __name__ == "__main__":
    
    cascade_file = "haarcascade_frontalface_alt (3).xml"  # Path to the classifier
    detect_faces_haarcascade_video(cascade_file)  # Calling the function to run detections
