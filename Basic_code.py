import cv2
import numpy as np

# Load the image
image = cv2.imread("C:/Users/ankit/OneDrive/Documents/Personal/Documents/Personal/My Photo.jpg")  # Replace with your image path
if image is None:
    print("Image not found. Please check the path.")
    exit()

# Resize the image for consistency
image = cv2.resize(image, (300, 300))

# Rotate 90 degrees clockwise
rotated = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)

# Grayscale conversion
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
gray_bgr = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

# HSV conversion
hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
hsv_bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

# Flipping horizontally
flipped = cv2.flip(image, 1)  # 1 means horizontal flip

# Add labels
def put_label(img, text, pos=(10, 25)):
    return cv2.putText(img, text, pos, cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

# Apply labels to all images
image_lbl = put_label(image.copy(), "Original")
rotated_lbl = put_label(rotated.copy(), "Rotated")
gray_lbl = put_label(gray_bgr.copy(), "Grayscale")
hsv_lbl = put_label(hsv_bgr.copy(), "HSV")
flipped_lbl = put_label(flipped.copy(), "Flipped")

# Stack images
row1 = np.hstack((image_lbl, rotated_lbl))
row2 = np.hstack((gray_lbl, hsv_lbl))
row3 = np.hstack((flipped_lbl, np.zeros_like(flipped_lbl)))  # Empty placeholder to align layout

# Combine all rows
combined = np.vstack((row1, row2, row3))

# Show the result
cv2.imshow("Image Transformations - OpenCV", combined)
cv2.waitKey(0)
cv2.destroyAllWindows()
