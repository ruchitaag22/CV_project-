import cv2

image = cv2.imread("sample.jpg")  
resized_image = cv2.resize(image, (400, 300))  
grayscale_image = cv2.cvtColor(resized_image, cv2.COLOR_BGR2GRAY)  

cv2.imshow("Grayscale Image", grayscale_image)  
cv2.imshow("Resized Image", resized_image)      
cv2.waitKey(0)  
cv2.destroyAllWindows()  
