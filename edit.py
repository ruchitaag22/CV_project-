import cv2

#Load image
image = cv2.imread("C:/Users/Acer/OneDrive/Documents/Open-CV Workshop/Day 1/sample.jpg")

#resize image
resized_image = cv2.resize(image, (800, 600))

#rotate image
rotated_image = cv2.rotate(resized_image, cv2.ROTATE_90_CLOCKWISE)

#cropped image
cropped_image = rotated_image[50:400, 100:500]

#flipped image
flipped_image = cv2.flip(cropped_image, 1)

cv2.imshow('flipped_image', flipped_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
