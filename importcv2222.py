import cv2
import numpy as np

image=cv2.imread("sample.jpg")
if image is None:
    print("Image not found, please check the path")
    exit()

resized_image = cv2.resize(image, (300, 500))
#cv2.imshow('Resized image', resized_image)

average_blurred_image = cv2.blur(image, (8, 8))
#cv2.imshow('Average Blurred Image', average_blurred_image)

gaussian_blurred_image = cv2.GaussianBlur(image, (9, 9), 0)
#cv2.imshow('Gaussian Blurred Image', gaussian_blurred_image)

median_blurred_image = cv2.medianBlur(image, 7)
#cv2.imshow('Median Blurred Image', median_blurred_image)

bilateral_blurred_image = cv2.bilateralFilter(image, 9, 75, 75)
#cv2.imshow('Bilateral Blurred Image', bilateral_blurred_image)

def put_label(img, text, pos=(10, 25)):
    return cv2.putText(img, text, pos, cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

original_lbl = put_label(resized_image.copy(), "Original Image")
avg_lbl = put_label(average_blurred_image.copy(), "Average Blurred Image")
gauss_lbl = put_label(gaussian_blurred_image.copy(), "Gussian Blurred Image")
median_lbl = put_label(median_blurred_image.copy(), "Median Blurred Image")
bilateral_lbl = put_label(bilateral_blurred_image.copy(), "Bilateral Blurred Image")

row1 = np.hstack((original_lbl, avg_lbl, gauss_lbl))
row2 = np.hstack((median_lbl, bilateral_lbl, np.zeros_like(image)))

final_image = np.stack((row1, row2))
cv2.imshow('Blurred Images Comparison', final_image)

cv2.waitKey(0)
cv2.destroyAllWindows()