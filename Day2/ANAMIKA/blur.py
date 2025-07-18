import cv2
import numpy as np

image= cv2.imread('sample.jpg')
if image is None:
    print("Error loading image")
    exit()

resized_image= cv2.resize(image, (300, 300))
#cv2.imshow('Resized Image', resized_image)

average_blurred_image = cv2.blur(resized_image, (8, 8))
#cv2.imshow('Average Blurred Image', average_blurred_image)

gaussian_blurred_image = cv2.GaussianBlur(resized_image, (9, 9), 0)
#cv2.imshow('Gaussian Blurred Image', gaussian_blurred_image)

median_blurred_image = cv2.medianBlur(resized_image, 5)
#cv2.imshow('Median Blurred Image', median_blurred_image)

bilateral_blurred_image = cv2.bilateralFilter(resized_image, 9, 75, 75)
#cv2.imshow('Bilateral Blurred Image', bilateral_blurred_image)  

def put_label(img,text,pos =(10,25)):
    return cv2.putText(img,text,pos, cv2.FONT_ITALIC , 0.6 ,(123,234,100), 3)

o_lbl= put_label(resized_image.copy(), 'Original Image')
avg_lbl= put_label(average_blurred_image.copy(), 'Average Blurred Image')
gaussian_lbl= put_label(gaussian_blurred_image.copy(), 'Gaussian Blurred Image')
median_lbl= put_label(median_blurred_image.copy(), 'Median Blurred Image')
bilateral_lbl= put_label(bilateral_blurred_image.copy(), 'Bilateral Blurred Image')

col1= np.vstack((o_lbl, avg_lbl))
col2= np.vstack((gaussian_lbl, median_lbl))
col3= np.vstack((bilateral_lbl, np.zeros_like(bilateral_lbl)))
final_image = np.hstack((col1, col2, col3))
cv2.imshow('Blurred Images Comparison', final_image)

cv2.waitKey(0)
cv2.destroyAllWindows()
