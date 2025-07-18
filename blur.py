import cv2


#rezize image
image = cv2.imread("C:/Users/Acer/OneDrive/Documents/Open-CV Workshop/Day 1/sample.jpg")
resized_image = cv2.resize(image, (800, 600))

#blur image
averageblur_image = cv2.blur(resized_image, (15, 15))

#gaussian blur image
gaussianblur_image = cv2.GaussianBlur(resized_image, (15, 15), 0)

#median blur image
medianblur_image = cv2.medianBlur(resized_image, 15)

#bilateral filter image
bilateralfilter_image = cv2.bilateralFilter(resized_image, 15, 75, 75)

cv2.imshow('averageblur_image', averageblur_image)
cv2.imshow('gaussianblur_image', gaussianblur_image)
cv2.imshow('medianblur_image', medianblur_image)
cv2.imshow('bilateral_image', bilateralfilter_image)
cv2.waitKey(0)
cv2.destroyAllWindows()


def put_label(img, text, pos=(10,25)):
    return cv2.putText(img, text, pos, cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

#Add labels
original_lbl = put_label(image.copy(), "Original")
avg_lbl = put_label(averageblur_image.copy(), "Average Blur")
gauss_lbl = put_label(gaussianblur_image.copy(), "Gaussian Blur")
median_lbl = put_label(medianblur_image.copy(), "Median Blur")
bilateralfilter_label = put_label(bilateralfilter_image.copy(), "Bilateral Filter")