import cv2

# Load and convert to grayscale
image = cv2.imread('sample.jpg')
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


# Apply simple binary threshold
_, thresh = cv2.threshold(gray, 170, 255, cv2.THRESH_BINARY)

# Show results
#cv2.imshow('Grayscale', gray)
cv2.imshow('Thresholded', thresh)
cv2.waitKey(0)
cv2.destroyAllWindows()