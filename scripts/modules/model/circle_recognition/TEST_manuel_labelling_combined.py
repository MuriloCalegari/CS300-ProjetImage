import os
import cv2 as cv
import csv
from modules.model.circle_detection.hough_transform import detect_circles, extract_hog_features, detect_cicles_opencv

def label_pieces(image_folder, output_file):
    with open(output_file, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['label', 'diameter', 'hog_features'])

        for image_file in os.listdir(image_folder):
            if image_file.endswith('.jpg'):
                image_path = os.path.join(image_folder, image_file)
                image = cv.imread(image_path)
                if image is None:
                    continue
                diameters = detect_cicles_opencv(image_path)

                if diameters is not None:
                    scale_percent = 50 
                    width = int(image.shape[1] * scale_percent / 100)
                    height = int(image.shape[0] * scale_percent / 100)
                    dim = (width, height)
                    resized_image = cv.resize(image, dim, interpolation=cv.INTER_AREA)

                    for diameter in diameters:
                        x, y, r = diameter[:3]
                        x, y, r = int(x * scale_percent / 100), int(y * scale_percent / 100), int(r * scale_percent / 100)
                        crop = resized_image[y-r:y+r, x-r:x+r]
                        hog_features, _ = extract_hog_features(crop)
                        hog_features_str = ','.join(map(str, hog_features))
                        label = input(f"Enter the label for this piece (diameter: {2*r}): ")
                        
                        writer.writerow([label, 2*r, hog_features_str])

train_image_folder = '/Volumes/SSD/ProjetImage/ProjetImage/DividedDataset/testset'

output_csv = '/Volumes/SSD/ProjetImage/ProjetImage/DividedDataset/labeled_data_test.csv'

# Appel de la fonction
label_pieces(train_image_folder, output_csv)

print("Les données ont été enregistrées dans le fichier", output_csv)
