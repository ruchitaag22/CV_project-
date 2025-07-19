import cv2

# Load image
image = cv2.imread('sample.jpg')

# Convert to grayscale and HSV
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

# Display results
cv2.imshow('Original', image)
cv2.imshow('Grayscale', gray)
cv2.imshow('HSV', hsv)
cv2.waitKey(0)
cv2.destroyAllWindows()