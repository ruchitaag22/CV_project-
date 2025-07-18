import cv2
import numpy as np

# Load the image
image = cv2.imread("C:/Users/ankit/OneDrive/Documents/Personal/Documents/Personal/My Photo.jpg")  # Replace with your image path
if image is None:
    print("Image not found. Please check the path.")
    exit()

# Resize for consistency
image = cv2.resize(image, (300, 300))

# Apply different types of blurs
blur_average = cv2.blur(image, (7, 7))                  # Averaging
blur_gaussian = cv2.GaussianBlur(image, (7, 7), 0)      # Gaussian Blur
blur_median = cv2.medianBlur(image, 7)                  # Median Blur
blur_bilateral = cv2.bilateralFilter(image, 15, 75, 75) # Bilateral Filter

# Function to label images
def put_label(img, text, pos=(10, 25)):
    return cv2.putText(img, text, pos, cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

# Add labels
original_lbl = put_label(image.copy(), "Original")
avg_lbl = put_label(blur_average.copy(), "Average Blur")
gauss_lbl = put_label(blur_gaussian.copy(), "Gaussian Blur")
median_lbl = put_label(blur_median.copy(), "Median Blur")
bilateral_lbl = put_label(blur_bilateral.copy(), "Bilateral Filter")

# Stack for display
row1 = np.hstack((original_lbl, avg_lbl, gauss_lbl))
row2 = np.hstack((median_lbl, bilateral_lbl, np.zeros_like(image)))  # padding for alignment

final = np.vstack((row1, row2))

# Display the result
cv2.imshow("Different Types of Blurring - OpenCV", final)
cv2.waitKey(0)
cv2.destroyAllWindows()
