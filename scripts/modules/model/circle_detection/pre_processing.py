import cv2 as cv

def apply_laplace(gray):
    ddepth = cv.CV_16S
    kernel_size = 3
    gray = cv.convertScaleAbs(cv.Laplacian(gray, ddepth, ksize=kernel_size))
    return gray