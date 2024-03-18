import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

from skimage.transform import hough_circle, hough_circle_peaks
from skimage.draw import circle_perimeter

from skimage import data, color, img_as_ubyte
from skimage.feature import canny
from skimage.transform import hough_ellipse
from skimage.draw import ellipse_perimeter

import cv2 as cv

# Load picture and detect edges
def detect_circles(image_path):
    """
    Detects circles in an image using the Hough Transform algorithm.
    
    Parameters:
        image_path (str): The path to the image file.

    Returns:
        set: A set of tuples (center, radius) where center is also a tuple (x, y).
    """
    return detect_cicles_opencv(image_path)

def detect_cicles_opencv(image_path):
    src = cv.imread(image_path, cv.IMREAD_COLOR)
    # Check if image is loaded fine
    if src is None:
        print ('Error opening image!')
        return -1
    
    ## Rescale the image to fit in a 2048x2048 window or keep the original size if it's smaller
    scale = 1
    if src.shape[0] > 1024 or src.shape[1] > 1024:
        scale = 1024 / max(src.shape[0], src.shape[1])
        src = cv.resize(src, (int(src.shape[1] * scale), int(src.shape[0] * scale)))
    
    gray = cv.cvtColor(src, cv.COLOR_BGR2GRAY)
    
    gray = cv.medianBlur(gray, 5)  
    
    rows = gray.shape[0]
    circles = cv.HoughCircles(gray, cv.HOUGH_GRADIENT, 1, rows / 8)

    output = set()
    
    if circles is not None:
        circles = np.uint16(np.around(circles))
        for i in circles[0, :]:
            print(f"Found circle! {i}")
            center = (i[0], i[1])
            # circle center
            cv.circle(src, center, 1, (0, 100, 100), 3)
            # circle outline
            radius = i[2]
            output.add(((int(i[0] * 1 / scale), int(i[1] * 1 / scale)), int(radius * 1 / scale)))
            cv.circle(src, center, radius, (255, 0, 255), 3)
    
    cv.imshow("detected circles", src)
    cv.waitKey(0)

    return output


detect_circles("/home/yassfkh/Desktop/ProjetImage/ProjetImage/Images/283.jpg")

