import cv2
import numpy as np

# Load and resize image
image = cv2.imread("C:/Users/ankit/OneDrive/Documents/Personal/Documents/Personal/My Photo.jpg")  # Replace with your image path
if image is None:
    print("Image not found. Please check the path.")
    exit()

image = cv2.resize(image, (500, 400))
annotated = image.copy()

# Draw a rectangle (top-left to bottom-right corner)
cv2.rectangle(annotated, (50, 50), (200, 150), (0, 255, 0), 60)

# Draw a circle (center, radius)
cv2.circle(annotated, (350, 100), 50, (255, 0, 0), 3)

# Draw a line (start point, end point)
cv2.line(annotated, (50, 300), (450, 300), (0, 0, 255), 2)

# Draw an arrowed line
cv2.arrowedLine(annotated, (100, 350), (400, 350), (255, 255, 0), 3, tipLength=0.1)

# Add text annotation
cv2.putText(annotated, 'Rectangle', (50, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
cv2.putText(annotated, 'Circle', (320, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
cv2.putText(annotated, 'Line', (230, 290), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
cv2.putText(annotated, 'Arrow', (230, 340), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

# Show result
cv2.imshow("Annotations and Shapes - OpenCV", annotated)
cv2.waitKey(0)
cv2.destroyAllWindows()
