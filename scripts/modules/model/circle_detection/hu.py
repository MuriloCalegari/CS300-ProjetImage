import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import skimage.feature as skf

def calculate_hu_moments(image):
    image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    hu_moments = cv.HuMoments(cv.moments(image)).flatten()
    return hu_moments

def display_image_and_hu_moments_difference(image1, image2, hu_moments1, hu_moments2, diff):
    fig, ax = plt.subplots(1, 3, figsize=(15, 5))

    ax[0].imshow(cv.cvtColor(image1, cv.COLOR_BGR2RGB))
    ax[0].set_title('Image 1')
    ax[0].axis('off')

    ax[1].imshow(cv.cvtColor(image2, cv.COLOR_BGR2RGB))
    ax[1].set_title('Image 2')
    ax[1].axis('off')

    ax[2].bar(np.arange(len(hu_moments1)), diff)
    ax[2].set_title('Difference in Hu Moments')
    ax[2].set_xlabel('Moment Index')
    ax[2].set_ylabel('Difference')

    plt.show()
    print("Difference in Hu Moments:", diff)

image_path1 = '/home/yassfkh/Desktop/ProjetImage/ProjetImage/scripts/cache/training_set/5_centimes/14_3_5_centimes.JPG'
image_path2 = '/home/yassfkh/Desktop/ProjetImage/ProjetImage/scripts/cache/training_set/2_euro/11_1_2_euro.JPG'
image1 = cv.imread(image_path1)
image2 = cv.imread(image_path2)



hu_moments1 = calculate_hu_moments(image1)
hu_moments2 = calculate_hu_moments(image2)

diff = np.abs(hu_moments1 - hu_moments2)

display_image_and_hu_moments_difference(image1, image2, hu_moments1, hu_moments2, diff)
