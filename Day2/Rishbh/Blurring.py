import cv2
import numpy as np

image = cv2.imread("Daisy_img.webp")
if image is None :
    print("Image Not Found! , Check the Path")
    exit()

resize_image = cv2.resize(image , (300 , 300))

average_blur = cv2.blur(resize_image , (7,7))
gaussian_blur = cv2.GaussianBlur(resize_image , (7, 7) , 0)
median_blur = cv2.medianBlur(resize_image, 7)
bilateral_blur = cv2.bilateralFilter(resize_image , 9 , 75 , 75)

def putlabel(img , text , pos =(10 ,25)):
    return cv2.putText(img , text , pos , cv2.FONT_HERSHEY_COMPLEX , 0.6 , (0 , 255 , 0 ) , 2)

original_image = putlabel(average_blur.copy() , "Average Blur Image")
Gaussian_image = putlabel(gaussian_blur.copy() , "Gaussian Image")
Median_blur = putlabel(median_blur.copy() , "Median Blur")
Bilateral_blur = putlabel(bilateral_blur.copy() , "Bilateral Filter")

rows = np.hstack((original_image , Gaussian_image))
rows1 = np.hstack((Median_blur , Bilateral_blur))

final = np.hstack((rows , rows1 ))

cv2.imshow( 'Blurred Matrix' ,  final)
cv2.waitKey(0)
cv2.destroyAllWindows()


