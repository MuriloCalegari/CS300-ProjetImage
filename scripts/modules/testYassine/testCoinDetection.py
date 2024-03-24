import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt



def display_image(image, title="Image"):
    plt.imshow(cv.cvtColor(image, cv.COLOR_BGR2RGB))
    plt.title(title)
    plt.axis('off')
    plt.show()

def resize_image(image, width, height):
    resized_image = cv.resize(image, (width, height))
   
    return resized_image

def RGB_to_greyscale(image):
    greyImage = cv.cvtColor(image, cv.COLOR_RGB2GRAY)
    
    return greyImage

def blur_image(image, ksize=19):
    blurredImage = cv.medianBlur(image, ksize) 
   
    return blurredImage


def laplace_filter(image):
    laplaceImage = cv.Laplacian(image,-1, ksize=5)
 
    return laplaceImage


def otsu_thresholding(channel):
    _, thresh = cv.threshold(channel, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)
    return thresh

def circular_hough_transform(image, radius_range, search_threshold):
    circles = cv.HoughCircles(image, cv.HOUGH_GRADIENT, dp=1, minDist=20, param1=50, param2=30, minRadius=radius_range[0], maxRadius=radius_range[1])
    if circles is not None:
        circles = np.uint16(np.around(circles))
        for i in circles[0, :]:
            center = (i[0], i[1])
            cv.circle(image, center, i[2], (0, 255, 0), 2)
            cv.circle(image, center, 2, (0, 0, 255), 3)

    return circles

image = cv.imread("/home/yassfkh/Desktop/ProjetImage/ProjetImage/Images/275.png", 1)



resized_image = resize_image(image, 400, 400)
greyImage = RGB_to_greyscale(resized_image)
blurredImage = blur_image(greyImage)

# NORMALISER APRES LE FLOU ET AVANT LAPLACE ?


laplaceImage = laplace_filter(blurredImage)
thresholdedImage = otsu_thresholding(laplaceImage)
thresholdedImageBeforeLaplace = otsu_thresholding(blurredImage)

plt.figure(figsize=(12, 8))

plt.subplot(3, 3, 1)
plt.imshow(cv.cvtColor(image, cv.COLOR_BGR2RGB))
plt.title("Image originale")
plt.axis('off')

plt.subplot(2, 3, 2)
plt.imshow(cv.cvtColor(resized_image, cv.COLOR_BGR2RGB))
plt.title("Resized image")
plt.axis('off')

plt.subplot(2, 3, 3)
plt.imshow(greyImage, cmap='gray')
plt.title("Greyscale image")
plt.axis('off')

plt.subplot(2, 3, 4)
plt.imshow(blurredImage, cmap='gray')
plt.title("Blurred image")
plt.axis('off')

plt.subplot(2, 3, 5)
plt.imshow(laplaceImage, cmap='gray')
plt.title("laplace image")
plt.axis('off')

plt.subplot(2, 3, 6)
plt.imshow(thresholdedImage, cmap='gray')
plt.title("Thresholded Image")
plt.axis('off')

plt.subplot(3,3, 4)
plt.imshow(thresholdedImageBeforeLaplace, cmap='gray')
plt.title("Thresholded before lapalce Image")
plt.axis('off')

plt.tight_layout()
plt.show()
