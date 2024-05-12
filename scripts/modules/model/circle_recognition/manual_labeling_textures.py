import os
import cv2 as cv
import csv
from modules.model.circle_detection.hough_transform import detect_cicles_opencv, extract_hog_features, extract_color_features

def pad_features(features, target_length=2048):
    """ Pad the feature list to the target length with zeros. """
    return features + [0] * (target_length - len(features))

def label_pieces(image_folder, output_file):
    # Determine the maximum length of HOG features
    max_hog_length = 0
    for image_file in os.listdir(image_folder):
        if image_file.endswith('.jpg'):
            image_path = os.path.join(image_folder, image_file)
            image = cv.imread(image_path)
            diameters = detect_cicles_opencv(image_path)
            for diameter in diameters:
                x, y, r = diameter[:3]
                crop = image[y - r:y + r, x - r:x + r]
                hog_features, _ = extract_hog_features(crop)
                if len(hog_features) > max_hog_length:
                    max_hog_length = len(hog_features)

    # Prepare the output CSV file
    with open(output_file, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['label', 'diameter', 'color_l_mean', 'color_a_mean', 'color_b_mean', 'hog_features'])

        # Process each image
        for image_file in os.listdir(image_folder):
            if image_file.endswith('.jpg'):
                image_path = os.path.join(image_folder, image_file)
                image = cv.imread(image_path)
                if image is None:
                    continue

                diameters = detect_cicles_opencv(image_path)
                if diameters is not None:
                    for diameter in diameters:
                        x, y, r = diameter[:3]
                        crop = image[y - r:y + r, x - r:x + r]

                        # Extract color features
                        color_features = extract_color_features(image_path, [diameter])
                        l_mean, a_mean, b_mean = color_features  # Assumes color_features returns a tuple

                        # Extract and pad HOG features
                        hog_features, _ = extract_hog_features(crop)
                        hog_features_padded = pad_features(hog_features, max_hog_length)
                        hog_features_str = ','.join(map(str, hog_features_padded))

                        # Get the label from the user
                        label = input(f"Enter the label for this piece (diameter: {2*r}): ")

                        # Write to CSV
                        writer.writerow([label, 2*r, l_mean, a_mean, b_mean, hog_features_str])

    print("Data has been saved to", output_file)

# Specify the folder and output file
train_image_folder = '/Volumes/SSD/ProjetImage/ProjetImage/DividedDataset/testset'
output_csv = '/Volumes/SSD/ProjetImage/ProjetImage/DividedDataset/labeled_data_test.csv'

# Execute the function
label_pieces(train_image_folder, output_csv)

print("The data has been saved to", output_csv)
