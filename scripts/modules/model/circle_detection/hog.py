import skimage.feature as skf
import skimage.filters as skfilt
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import skimage.feature as skf

def visualize_hog_features(image, cell_size=(8, 8), block_size=(2, 2), nbins=9):
    gray_image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    hog_features, hog_image = skf.hog(gray_image, orientations=nbins, pixels_per_cell=cell_size, cells_per_block=block_size, visualize=True)
    plt.figure(figsize=(10, 10))
    plt.imshow(hog_image, cmap='gray')
    plt.axis('off')
    plt.show()
    return hog_features

# Exemple d'image
image = cv.imread('/home/yassfkh/Desktop/ProjetImage/ProjetImage/scripts/cache/training_set/2_euro/116_0_2_euro.jpg')

# Visualisation des caractéristiques HOG et récupération du vecteur de texture
hog_features = visualize_hog_features(image)
print("HOG Features:", hog_features)
