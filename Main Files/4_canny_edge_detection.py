import cv2

# Load image
image = cv2.imread('sample.jpg')
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
blurred = cv2.GaussianBlur(gray, (5, 5), 0)

# Apply Canny edge detector
edges = cv2.Canny(blurred, 100, 200) #lower and upper thresholds for edge detection 200 = big difference in pixel values

# Show result
cv2.imshow('Edges', edges)
cv2.waitKey(0)
cv2.destroyAllWindows()