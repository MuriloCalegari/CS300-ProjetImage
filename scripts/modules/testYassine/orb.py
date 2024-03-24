import cv2 as cv
import numpy as np
import argparse

parser = argparse.ArgumentParser(description='Code for Feature Matching with FLANN tutorial.')
parser.add_argument('--input1', help='Path to input image 1.', default='/home/yassfkh/Desktop/ProjetImage/ProjetImage/scripts/cache/testing_set/5_centimes/212_1_5_centimes.jpg')
parser.add_argument('--input2', help='Path to input image 2.', default='/home/yassfkh/Desktop/ProjetImage/ProjetImage/scripts/cache/testing_set/20_centimes/44_0_20_centimes.jpg')
args = parser.parse_args()

img1 = cv.imread(cv.samples.findFile(args.input1), cv.IMREAD_GRAYSCALE)
img2 = cv.imread(cv.samples.findFile(args.input2), cv.IMREAD_GRAYSCALE)
if img1 is None or img2 is None:
    print('Could not open or find the images!')
    exit(0)

#-- Step 1: Detect the keypoints using ORB Detector, compute the descriptors
orb = cv.ORB_create()
keypoints1, descriptors1 = orb.detectAndCompute(img1, None)
keypoints2, descriptors2 = orb.detectAndCompute(img2, None)

#-- Step 2: Matching descriptor vectors with a FLANN based matcher
# Since ORB is a binary descriptor, NORM_HAMMING should be used
matcher = cv.DescriptorMatcher_create(cv.DescriptorMatcher_BRUTEFORCE_HAMMING)
matches = matcher.match(descriptors1, descriptors2)

#-- Draw matches
img_matches = np.empty((max(img1.shape[0], img2.shape[0]), img1.shape[1]+img2.shape[1], 3), dtype=np.uint8)
cv.drawMatches(img1, keypoints1, img2, keypoints2, matches, img_matches, flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

#-- Show detected matches
cv.imshow('Good Matches', img_matches)

cv.waitKey()
