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
    rows = image.shape[0]
    circles = cv.HoughCircles(image, cv.HOUGH_GRADIENT, 1, rows / 8)
    if circles is not None:
        circles = np.uint16(np.around(circles))
        for i in circles[0, :]:
            center = (i[0], i[1])
            cv.circle(image, center, i[2], (0, 255, 0), 2)
            cv.circle(image, center, 2, (0, 0, 255), 3)

    return circles


def apply_closing(image, kernel_size):
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    closing = cv.morphologyEx(image, cv.MORPH_CLOSE, kernel)
    return closing

def apply_opening(image, kernel_size):
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    opening = cv.morphologyEx(image, cv.MORPH_OPEN, kernel)
    return opening


def normalize_image(image):
    normalized_image = np.zeros(image.shape, np.uint8)
    cv.normalize(image, normalized_image, 0, 255, cv.NORM_MINMAX)
    return normalized_image


image = cv.imread("/home/yassfkh/Desktop/ProjetImage/ProjetImage/Images/211.jpg", 1)



resized_image = resize_image(image, 400, 400)
greyImage = RGB_to_greyscale(resized_image)
normalized_blurredImage = normalize_image(greyImage)
blurredImage = blur_image(normalized_blurredImage)
laplaceImage = laplace_filter(blurredImage)
thresholdedImage = otsu_thresholding(laplaceImage)
thresholdedImageBeforeLaplace = otsu_thresholding(blurredImage)

closedImage = apply_closing(thresholdedImage, kernel_size=6)
openImage = apply_opening(thresholdedImage, kernel_size=6)

plt.figure(figsize=(18, 12))

plt.subplot(3, 4, 1)
plt.imshow(cv.cvtColor(image, cv.COLOR_BGR2RGB))
plt.title("Image originale")
plt.axis('off')

plt.subplot(3, 4, 2)
plt.imshow(cv.cvtColor(resized_image, cv.COLOR_BGR2RGB))
plt.title("Resized image")
plt.axis('off')

plt.subplot(3, 4, 3)
plt.imshow(greyImage, cmap='gray')
plt.title("Greyscale image")
plt.axis('off')

plt.subplot(3, 4, 4)
plt.imshow(blurredImage, cmap='gray')
plt.title("Blurred image")
plt.axis('off')

plt.subplot(3, 4, 5)
plt.imshow(laplaceImage, cmap='gray')
plt.title("Laplace image")
plt.axis('off')

plt.subplot(3, 4, 6)
plt.imshow(thresholdedImage, cmap='gray')
plt.title("Thresholded Image")
plt.axis('off')

plt.subplot(3, 4, 7)
plt.imshow(thresholdedImageBeforeLaplace, cmap='gray')
plt.title("Thresholded before Laplace Image")
plt.axis('off')

plt.subplot(3, 4, 8)
plt.imshow(closedImage, cmap='gray')
plt.title("Closed Image")
plt.axis('off')

plt.subplot(3, 4, 9)
plt.imshow(openImage, cmap='gray')
plt.title("Opened Image")
plt.axis('off')

plt.subplot(3, 4, 10)  
plt.imshow(normalized_blurredImage, cmap='gray') 
plt.title("Normalized Blurred Image")
plt.axis('off')

circles_detected = circular_hough_transform(blurredImage, radius_range=[20, 100], search_threshold=50)
image_for_display = cv.cvtColor(openImage, cv.COLOR_GRAY2BGR)

if circles_detected is not None:
    for circle in circles_detected[0, :]:
        center = (circle[0], circle[1])  
        radius = circle[2]  
        cv.circle(image_for_display, center, radius, (0, 255, 0), 2)  
        cv.circle(image_for_display, center, 2, (0, 0, 255), 3) 
        
print(circles_detected)


plt.figure(figsize=(6, 6))
plt.imshow(cv.cvtColor(image_for_display, cv.COLOR_BGR2RGB))
plt.title("Image with Detected Circles")
plt.axis('off')
plt.show()

plt.tight_layout()
plt.show()
