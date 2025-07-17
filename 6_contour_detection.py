import cv2

# Load and convert image
image = cv2.imread('sample.jpg')
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
_, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY) #cv2.threshold returns a tuple, we use _ to ignore the first value

# Find contours
contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE) #cv2.findContours returns a list of contours and hierarchy _ is used to ignore the hierarchy

# Draw contours on original image
output = image.copy()
cv2.drawContours(output, contours, -1, (0, 255, 255), 2)

# Show results
cv2.imshow('Original', gray)
cv2.imshow('Contours', output)
cv2.waitKey(0)
cv2.destroyAllWindows()