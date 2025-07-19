import cv2

# Load image
image = cv2.imread('sample.jpg')

# Show image
cv2.imshow('Original Image', image)
cv2.waitKey(0)
cv2.destroyAllWindows()