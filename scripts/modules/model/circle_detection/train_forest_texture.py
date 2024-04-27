import pandas as pd
from sklearn import svm
from sklearn.metrics import accuracy_score
import numpy as np
import ast
import cv2 as cv
from joblib import dump
from joblib import load

from modules.model.circle_detection.hough_transform import detect_circles, extract_hog_features, detect_cicles_opencv

""" def load_data_and_train_svm(train_file):
    train_data = pd.read_csv(train_file)
    
    train_data['hog_features'] = train_data['hog_features'].apply(ast.literal_eval)

    max_length = max(len(x) for x in train_data['hog_features'])
    X_train = np.array([xi + [0]*(max_length - len(xi)) if len(xi) < max_length else xi for xi in train_data['hog_features']]).astype(np.float32)
    
    y_train = train_data['label']

    svm_model = svm.SVC(kernel='linear', C=1.0)
    svm_model.fit(X_train, y_train)

    dump(svm_model, 'svm_model.joblib')

    return svm_model

train_file = '/Volumes/SSD/ProjetImage/ProjetImage/scripts/labeled_data_hog_features.csv'

svm_model = load_data_and_train_svm(train_file)
 """

svm_model = load('svm_model.joblib')

def predict_with_hog_features(image_path, svm_model):
    image = cv.imread(image_path)
    if image is None:
        print("Image non trouvée.")
        return

    scale_percent = 50
    width = int(image.shape[1] * scale_percent / 100)
    height = int(image.shape[0] * scale_percent / 100)
    dim = (width, height)
    resized_image = cv.resize(image, dim, interpolation=cv.INTER_AREA)

    diameters = detect_cicles_opencv(image_path)
    if diameters is not None and len(diameters) > 0:
        for diameter in diameters:
            x, y, r = diameter[:3]
            x, y, r = int(x * scale_percent / 100), int(y * scale_percent / 100), int(r * scale_percent / 100)
            crop_img = resized_image[y-r:y+r, x-r:x+r]
            if crop_img.size == 0:
                print("Région croppée vide.")
                continue

      
            hog_features, hog_image = extract_hog_features(crop_img)
            hog_features_padded = pad_features(hog_features)
            hog_features_padded = np.array(hog_features_padded).reshape(1, -1) 
            prediction = svm_model.predict(hog_features_padded)
            print("Prédiction pour la pièce détectée :", prediction)



def pad_features(hog_features, target_length=207936):
    padding_size = target_length - len(hog_features)
    if padding_size > 0:
        padding = np.zeros(padding_size)

        hog_features_padded = np.concatenate([hog_features, padding])
    else:
        hog_features_padded = hog_features

    return hog_features_padded



test_image_path = '/Volumes/SSD/ProjetImage/ProjetImage/DividedDataset/valset/28.jpg'
predict_with_hog_features(test_image_path, svm_model)