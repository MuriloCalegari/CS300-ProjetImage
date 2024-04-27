import os
import cv2 as cv
import csv
from modules.model.circle_detection.hough_transform import detect_cicles_opencv, extract_color_features

def label_pieces(image_folder, output_file):

    with open(output_file, mode='a', newline='') as file:
        writer = csv.writer(file)
        
        writer.writerow(['label', 'diameter', 'color_l_mean', 'color_a_mean', 'color_b_mean'])

        for image_file in os.listdir(image_folder):
            if image_file.endswith('.jpg'):
                image_path = os.path.join(image_folder, image_file)
                diameters = detect_cicles_opencv(image_path)

                if diameters is not None:
                    image = cv.imread(image_path)
                    scale_percent = 50 
                    width = int(image.shape[1] * scale_percent / 100)
                    height = int(image.shape[0] * scale_percent / 100)
                    dim = (width, height)
                    resized_image = cv.resize(image, dim, interpolation=cv.INTER_AREA)

                    for diameter in diameters:
                        x, y, r = diameter[:3]
                        x, y, r = int(x * scale_percent / 100), int(y * scale_percent / 100), int(r * scale_percent / 100)
                        image_with_circle = resized_image.copy()
                        cv.circle(image_with_circle, (x, y), r, (0, 255, 0), 2)
                        cv.imshow('Image', image_with_circle)
                        cv.waitKey(0)
                        label = input(f"Enter the label for this piece (diameter: {diameter[3]} px): ")
                        
                  
                        features = extract_color_features(image_path, [diameter])
                        diameter, l_mean, a_mean, b_mean = features[0]

                      
                        writer.writerow([label, diameter, l_mean, a_mean, b_mean])
                        cv.destroyAllWindows()

train_image_folder = '/Volumes/SSD/ProjetImage/ProjetImage/DividedDataset/testset'
output_csv = '/Volumes/SSD/ProjetImage/ProjetImage/DividedDataset/labeled_data_test.csv' 

label_pieces(train_image_folder, output_csv)

print("Les données ont été enregistrées dans le fichier", output_csv)
