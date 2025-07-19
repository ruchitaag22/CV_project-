import cv2

# Load and convert to grayscale
image = cv2.imread('sample.jpg')
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Apply Gaussian Blur
blurred = cv2.GaussianBlur(gray, (5, 5), 0) #kernel size (5, 5) must be odd number so kernel grid has a center pixel

# Display images
cv2.imshow('Original', gray)
cv2.imshow('Blurred', blurred)
cv2.waitKey(0)
cv2.destroyAllWindows()