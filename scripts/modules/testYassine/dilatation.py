import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

image = np.zeros((256, 256), dtype=np.uint8)

cv.circle(image, (64, 64), 30, 255, -1)
cv.circle(image, (128, 128), 40, 255, -1)
cv.circle(image, (192, 192), 50, 255, -1)

_, binary_image = cv.threshold(image, 127, 255, cv.THRESH_BINARY)

kernel = np.ones((5, 5), np.uint8)
dilated_image = cv.dilate(binary_image, kernel, iterations=1)

eroded_image = cv.erode(binary_image, kernel, iterations=1)

closing_image = cv.morphologyEx(binary_image, cv.MORPH_CLOSE, kernel)    
opening_image = cv.morphologyEx(binary_image, cv.MORPH_OPEN, kernel)


plt.figure(figsize=(10, 12)) 

plt.subplot(3, 2, 1)
plt.imshow(binary_image, cmap='gray')
plt.title('Binary Image')
plt.axis('off')

plt.subplot(3, 2, 2)
plt.imshow(dilated_image, cmap='gray')
plt.title('Dilated Image')
plt.axis('off')

plt.subplot(3, 2, 3)
plt.imshow(eroded_image, cmap='gray')
plt.title('Eroded Image')
plt.axis('off')

plt.subplot(3, 2, 4)
plt.imshow(closing_image, cmap='gray')
plt.title('Closing Image')
plt.axis('off')

plt.subplot(3, 2, 5)
plt.imshow(opening_image, cmap='gray')
plt.title('Opening Image') 
plt.axis('off')  


plt.tight_layout()
plt.show()
