import numpy as np
import cv2 as cv
from joblib import load
import json
from modules.utils import get_parameter, get_parameters
from modules.model.circle_detection.hough_transform import detect_circles, extract_hog_features, detect_cicles_opencv
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
from collections import Counter

mapping = {
    "50cts": "50_centimes",
    "20cts": "20_centimes",
    "10cts": "10_centimes",
    "5cts": "5_centimes",
    "2cts": "2_centimes",
    "1cts": "1_centimes",
    "1e": "1_euro",
    "2e": "2_euro"
}

def predict_with_hog_features(image_path, svm_model):
    image = cv.imread(image_path)
    if image is None:
        print("Image non trouvée.")
        return None

    scale_percent = 50
    width = int(image.shape[1] * scale_percent / 100)
    height = int(image.shape[0] * scale_percent / 100)
    dim = (width, height)
    resized_image = cv.resize(image, dim, interpolation=cv.INTER_AREA)


    predictions = []  

    diameters = detect_cicles_opencv(image_path)
    if diameters is not None and len(diameters) > 0:
        for diameter in diameters:
            if len(diameter) == 3:
                x, y, r = diameter[:3]
            elif len(diameter) == 2:
                (x, y), r = diameter
            else:
                print("Invalid diameter tuple length")
                continue
            
            x, y, r = int(x * scale_percent / 100), int(y * scale_percent / 100), int(r * scale_percent / 100)
            crop_img = resized_image[y-r:y+r, x-r:x+r]
            if crop_img.size == 0:
                print("Région croppée vide.")
                continue

            hog_features, hog_image = extract_hog_features(crop_img)
            hog_features_padded = pad_features(hog_features)
            hog_features_padded = np.array(hog_features_padded).reshape(1, -1)
            prediction = svm_model.predict(hog_features_padded)
            label = str(prediction[0])

            label = mapping.get(label, label)

            predictions.append((label, (x, y), r))  

            text = label
            cv.putText(resized_image, text, (x, y), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv.LINE_AA)

    cv.imshow('Detected Coins with Predictions', resized_image)
    cv.waitKey(0)
    cv.destroyAllWindows()

    return predictions


def pad_features(hog_features, target_length=207936):
    padding_size = target_length - len(hog_features)
    if padding_size > 0:
        padding = np.zeros(padding_size)
        hog_features_padded = np.concatenate([hog_features, padding])
    else:
        hog_features_padded = hog_features

    return hog_features_padded

path = get_parameters()["svm_model_path"]

svm_model = load(path)

""" test_image_path = '/Volumes/SSD/ProjetImage/ProjetImage/Images/218.jpg'
predictions = predict_with_hog_features(test_image_path, svm_model)

for prediction in predictions:
    print(prediction)
 """