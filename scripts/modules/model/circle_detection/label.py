import os
import csv
import cv2 as cv
import numpy as np
from modules.model.circle_detection.hough_transform import detect_cicles_opencv, extract_color_and_hog_features


def label_pieces(image_folder, output_file):
    labeled_data = []

    for image_file in os.listdir(image_folder):
        if image_file.endswith('.jpg'):
            image_path = os.path.join(image_folder, image_file)
            diameters = detect_cicles_opencv(image_path)

            if diameters is not None:
                image = cv.imread(image_path)
                scale_percent = 30  # %
                width = int(image.shape[1] * scale_percent / 100)
                height = int(image.shape[0] * scale_percent / 100)
                dim = (width, height)
                resized_image = cv.resize(image, dim, interpolation = cv.INTER_AREA)

                for diameter in diameters:
                    x, y, r = diameter[:3]
                    x, y, r = int(x * scale_percent / 100), int(y * scale_percent / 100), int(r * scale_percent / 100)
                    image_with_circle = resized_image.copy()
                    cv.circle(image_with_circle, (x, y), r, (0, 255, 0), 2)
                    cv.imshow('Image', image_with_circle)
                    cv.waitKey(0)
                    label = input("Entrez le label pour cette pièce (diamètre : {} px) : ".format(diameter[3]))
                    features = extract_color_and_hog_features(image_path, [diameter])
                    labeled_data.append((label, features))
                    cv.destroyAllWindows()  #




train_image_folder = '/home/yassfkh/Desktop/ProjetImage/DividedDataset/trainset'
output_csv = 'labeled_data.csv'

label_pieces(train_image_folder, output_csv)

print("Les données ont été enregistrées dans le fichier", output_csv)
