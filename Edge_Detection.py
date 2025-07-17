import cv2
import numpy as np

# Load the image
image = cv2.imread("C:/Users/ankit/OneDrive/Documents/Personal/Documents/Personal/My Photo.jpg")  # Replace with your image path
if image is None:
    print("Image not found. Please check the path.")
    exit()

# Resize for consistency
image = cv2.resize(image, (300, 300))

# Convert to grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# 1. Canny Edge Detection
edges_canny = cv2.Canny(gray, 50, 150)

# 2. Sobel Edge Detection (X and Y then combined)
sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)  # X direction
sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)  # Y direction
sobel_combined = cv2.magnitude(sobelx, sobely)
sobel_combined = np.uint8(np.clip(sobel_combined, 0, 255))

# 3. Laplacian Edge Detection
laplacian = cv2.Laplacian(gray, cv2.CV_64F)
laplacian = np.uint8(np.clip(laplacian, 0, 255))

# 4. Binary Thresholding (Binary Edge Detection)
_, binary = cv2.threshold(gray, 100, 255, cv2.THRESH_BINARY)

# 5. Adaptive Thresholding
adaptive = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                  cv2.THRESH_BINARY, 11, 2)

# Convert single channel to BGR for stacking
def to_bgr(img):
    return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

# Add label text
def put_label(img, text, pos=(10, 25)):
    return cv2.putText(img, text, pos, cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

# Prepare labeled images
original = put_label(image.copy(), "Original")
canny = put_label(to_bgr(edges_canny), "Canny")
sobel = put_label(to_bgr(sobel_combined), "Sobel")
lap = put_label(to_bgr(laplacian), "Laplacian")
binary = put_label(to_bgr(binary), "Binary Threshold")
adaptive = put_label(to_bgr(adaptive), "Adaptive Threshold")

# Stack images
row1 = np.hstack((original, canny, sobel))
row2 = np.hstack((lap, binary, adaptive))
final = np.vstack((row1, row2))

# Show result
cv2.imshow("Edge Detection Techniques - OpenCV", final)
cv2.waitKey(0)
cv2.destroyAllWindows()