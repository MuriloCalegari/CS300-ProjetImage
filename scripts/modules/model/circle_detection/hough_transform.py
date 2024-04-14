import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

from skimage.transform import hough_circle, hough_circle_peaks
from skimage.draw import circle_perimeter

from skimage import data, color, img_as_ubyte
from skimage.feature import canny
from skimage.transform import hough_ellipse
from skimage.draw import ellipse_perimeter

from skimage.feature import peak_local_max
from skimage.segmentation import watershed
from scipy import ndimage
import imutils

from modules.model.circle_detection.utils import remove_overlapping_circles
from modules.utils import get_parameter
from modules.model.circle_detection.pre_processing import apply_laplace
from modules.model.circle_detection.enchanced_hough_search import find_circles

import cv2 as cv
import math
import skimage.feature as skf

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

def detect_circles_watershed(image_path):
    src = cv.imread(image_path, cv.IMREAD_COLOR)
    # Check if image is loaded fine
    if src is None:
        print ('Error opening image!')
        return -1
    
    ## Rescale the image to fit in a max_res x max_res window or keep the original size if it's smaller
    scale = 1
    max_largest_dim = get_parameter("hough_parameters")["max_res"]
    if src.shape[0] > max_largest_dim or src.shape[1] > max_largest_dim:
        scale = max_largest_dim / max(src.shape[0], src.shape[1])
        src = cv.resize(src, (int(src.shape[1] * scale), int(src.shape[0] * scale)))

    shifted = cv.pyrMeanShiftFiltering(src, 21, 51)
    cv.imshow("Input", src)
    gray = cv.cvtColor(shifted, cv.COLOR_BGR2GRAY)
    thresh = cv.threshold(gray, 0, 255,
    cv.THRESH_BINARY | cv.THRESH_OTSU)[1]
    cv.imshow("Thresh", thresh)

    # compute the exact Euclidean distance from every binary
    # pixel to the nearest zero pixel, then find peaks in this
    # distance map
    D = ndimage.distance_transform_edt(thresh)
    localMax = peak_local_max(D, min_distance=20, labels=thresh)
    peaks_mask = np.zeros_like(D, dtype=bool)
    peaks_mask[localMax] = True

    localMax = peaks_mask

    # perform a connected component analysis on the local peaks,
    # using 8-connectivity, then apply the Watershed algorithm
    markers = ndimage.label(localMax, structure=np.ones((3, 3)))[0]
    labels = watershed(-D, markers, mask=thresh)
    print("[INFO] {} unique segments found".format(len(np.unique(labels)) - 1))

    # loop over the unique labels returned by the Watershed
    # algorithm
    for label in np.unique(labels):
        # if the label is zero, we are examining the 'background'
        # so simply ignore it
        if label == 0:
            continue
        # otherwise, allocate memory for the label region and draw
        # it on the mask
        mask = np.zeros(gray.shape, dtype="uint8")
        mask[labels == label] = 255
        # detect contours in the mask and grab the largest one
        cnts = cv.findContours(mask.copy(), cv.RETR_EXTERNAL,
            cv.CHAIN_APPROX_SIMPLE)
        cnts = imutils.grab_contours(cnts)
        c = max(cnts, key=cv.contourArea)
        # draw a circle enclosing the object
        ((x, y), r) = cv.minEnclosingCircle(c)
        cv.circle(src, (int(x), int(y)), int(r), (0, 255, 0), 2)
        cv.putText(src, "#{}".format(label), (int(x) - 10, int(y)),
            cv.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
    
    # show the output image
    cv.imshow("Output", src)
    cv.waitKey(0)


def denormalize_1d(length, image_shape):
    return length * math.sqrt(image_shape[0] * image_shape[1])

max_lowThreshold = 100
window_name = 'Edge Map'
title_trackbar = 'Min Threshold:'
ratio = 3
kernel_size = 3

def CannyThreshold(val, src, src_gray):
    low_threshold = val
    img_blur = cv.blur(src_gray, (3,3))
    detected_edges = cv.Canny(img_blur, low_threshold, low_threshold*ratio, kernel_size)
    mask = detected_edges != 0
    dst = src * (mask[:,:,None].astype(src.dtype))
    return dst

def detect_cicles_opencv(image_path):
    print(image_path)
    src = cv.imread(image_path, cv.IMREAD_COLOR)
    # Check if image is loaded fine
    if src is None:
        print ('Error opening image!')
        return -1

    ## Rescale the image to fit in a max_res x max_res window or keep the original size if it's smaller
    scale = 1
    max_largest_dim = get_parameter("hough_parameters")["max_res"]
    if src.shape[0] > max_largest_dim or src.shape[1] > max_largest_dim:
        scale = max_largest_dim / max(src.shape[0], src.shape[1])
        src = cv.resize(src, (int(src.shape[1] * scale), int(src.shape[0] * scale)))

    gray = cv.cvtColor(src, cv.COLOR_BGR2GRAY)

    gray = cv.medianBlur(gray, 5)
    # gray = cv.GaussianBlur(gray, (3, 3), 0)

    # Equalize image's histogram
    # gray = cv.equalizeHist(gray)

    # Apply Laplace
    if(get_hough_parameters().get("apply_laplace")):
        gray = apply_laplace(gray)
    # gray = cv.GaussianBlur(gray, (3, 3), 0)
    # cv.imshow("Laplace", gray)

    # Apply OTSU to the image
    # ret1, mask = cv.threshold(gray,0,255,cv.THRESH_BINARY+cv.THRESH_OTSU)
    # gray = cv.bitwise_and(gray, gray, mask=mask)

    # Apply opening
    # gray = cv.medianBlur(gray, 9)
    # kernel = np.ones((3,3),np.uint8)
    # gray = cv.morphologyEx(gray, cv.MORPH_OPEN, kernel)

    # Show grayscale image
    cv.imshow('Grayscale Image', cv.cvtColor(gray, cv.COLOR_GRAY2RGB))

    after_canny = CannyThreshold(0, src, gray)

    # Show Canny threshold image
    cv.imshow('After Canny', after_canny)

    rows = gray.shape[0]

    parameters = get_hough_parameters()

    if(get_parameter("hough_parameters")["use_default_hough"]):
        circles = cv.HoughCircles(gray, cv.HOUGH_GRADIENT, 1, rows / 8)
    else:
        min_radius = int(denormalize_1d(parameters["min_radius"], gray.shape))
        max_radius = int(denormalize_1d(parameters["max_radius"], gray.shape))
        minDist = int(denormalize_1d(parameters["minDist"], gray.shape))

        print(f"Running hough transform with parameters: {parameters}")

        print(f"Min radius: {min_radius}, Max radius: {max_radius}, minDist: {minDist}")
        circles = cv.HoughCircles(gray, cv.HOUGH_GRADIENT_ALT, parameters["dp"], minDist=minDist,
            param1=parameters["param1"], param2=parameters["param2"],
            minRadius=min_radius, maxRadius=max_radius)

    #output = set()
    output = []

    if circles is not None and circles.any():
        circles = np.uint16(np.around(circles[0, :]))

        if(get_parameter("hough_parameters")["post_processing"]["remove_overlapping"]):
            circles = remove_overlapping_circles(circles)

        for i in circles:
            center = (i[0], i[1])
            radius = i[2]
            diameter = 2 * radius
            scaled_center = (int(center[0] * 1 / scale), int(center[1] * 1 / scale))
            scaled_radius = int(radius * 1 / scale)
            output.append((scaled_center[0], scaled_center[1], scaled_radius, diameter))

            cv.circle(src, center, radius, (255, 0, 255), 3)

            text = f"{int(diameter)} mm"
            text_position = (int(i[0] - 20), int(i[1] - 20))
            cv.putText(src, text, text_position, cv.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            

    # Show detected circles
    cv.imshow('Detected Circles', src)

    if(get_parameter("hough_parameters")["show_preview"]):
        cv.waitKey(0)
        cv.destroyAllWindows()

    output = np.array(output)
    return output


def get_hough_parameters():
    parameters = get_parameter("hough_parameters")
    
    if(parameters.get("param1") is None):
        parameters["param1"] = 100
    if parameters.get("param2") is None:
        parameters["param2"] = 30
    if parameters.get("min_radius") is None:
        parameters["min_radius"] = 1
    if parameters.get("max_radius") is None:
        parameters["max_radius"] = 30
    if parameters.get("dp") is None:
        parameters["dp"] = 1
    if parameters.get("minDist") is None:
        parameters["minDist"] = 30

    return parameters
    

def detect_circles_skimage():
    image = img_as_ubyte(data.coins()[160:230, 70:270])
    edges = canny(image, sigma=3, low_threshold=10, high_threshold=50)


    # Detect two radii
    hough_radii = np.arange(20, 35, 2)
    hough_res = hough_circle(edges, hough_radii)

    # Select the most prominent 3 circles
    accums, cx, cy, radii = hough_circle_peaks(hough_res, hough_radii,
                                            total_num_peaks=3)

    # Draw them
    fig, ax = plt.subplots(ncols=1, nrows=1, figsize=(10, 4))
    image = color.gray2rgb(image)
    for center_y, center_x, radius in zip(cy, cx, radii):
        circy, circx = circle_perimeter(center_y, center_x, radius,
                                        shape=image.shape)
        image[circy, circx] = (220, 20, 20)

    ax.imshow(image, cmap=plt.cm.gray)
    plt.show()
    
    

def extract_color_and_hog_features(image_path, circles):
    src = cv.imread(image_path, cv.IMREAD_COLOR)
    features_list = []
    for (x, y, r, diameter) in circles:
        x1 = max(x - r, 0)
        y1 = max(y - r, 0)
        x2 = min(x + r, src.shape[1])
        y2 = min(y + r, src.shape[0])
        crop = src[y1:y2, x1:x2]
        
        if crop.size == 0:
            continue 

        lab_crop = cv.cvtColor(crop, cv.COLOR_BGR2LAB)
        l_mean = np.mean(lab_crop[:, :, 0])
        a_mean = np.mean(lab_crop[:, :, 1])
        b_mean = np.mean(lab_crop[:, :, 2])
        
        hog_features, hog_image = extract_hog_features(crop)  
        
        features_list.append([diameter, l_mean, a_mean, b_mean, hog_features.tolist()])
        
        plt.figure(figsize=(10, 10))
        plt.imshow(hog_image, cmap='gray')
        plt.title(f'HOG features for Circle at ({x}, {y})')
        plt.axis('off')
        plt.show()

    return features_list


def extract_hog_features(region, cell_size=(8, 8), block_size=(2, 2), nbins=9):
    gray_region = cv.cvtColor(region, cv.COLOR_BGR2GRAY)
    hog_features, hog_image = skf.hog(gray_region, orientations=nbins, pixels_per_cell=cell_size, cells_per_block=block_size, visualize=True)
    return hog_features, hog_image


def create_features_vector(image_path):
    circles = detect_cicles_opencv(image_path)
    if circles.size > 0:
        color_features = extract_color_and_hog_features(image_path, circles)
        return color_features
    else:
        return []