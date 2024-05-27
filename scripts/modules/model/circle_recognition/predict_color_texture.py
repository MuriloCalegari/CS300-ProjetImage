import numpy as np
import cv2 as cv
from joblib import load

from modules.model.circle_detection.hough_transform import detect_circles, extract_hog_features, detect_cicles_opencv


def pad_features(hog_features, target_length=853780):
    adjusted_length = target_length - 4
    padding_size = adjusted_length - len(hog_features)
    if padding_size > 0:
        padding = np.zeros(padding_size)
        hog_features_padded = np.concatenate([hog_features, padding])
    else:
        hog_features_padded = hog_features
    return hog_features_padded


def predict_with_combined_features(image_path, svm_model, scaler):
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

            # extraction et padding des caractéristiques hog
            hog_features, hog_image = extract_hog_features(crop_img)
            print("Length of HOG features before padding:", len(hog_features))
            hog_features_padded = pad_features(hog_features)
            print("Length of HOG features after padding:", len(hog_features_padded))
            hog_features_padded = np.array(hog_features_padded).reshape(1, -1)

            # extraction des caractéristiques de couleur et ajout au vecteur de caractéristiques
            lab_image = cv.cvtColor(crop_img, cv.COLOR_BGR2LAB)
            l, a, b = cv.split(lab_image)
            color_features = [np.mean(l), np.mean(a), np.mean(b)]
            diameter = 2 * r
            
            # concatenation de toutes les caractéristiques avant scaling
            all_features = np.concatenate(([diameter], color_features, hog_features_padded[0]))

            # scaling et prédiction
            all_features_scaled = scaler.transform([all_features])
            prediction = svm_model.predict(all_features_scaled)
            print("Prédiction pour la pièce détectée :", prediction)


            text = str(prediction[0])
            cv.putText(resized_image, text, (x, y), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv.LINE_AA)

    cv.imshow('Detected Coins with Predictions', resized_image)
    cv.waitKey(0)
    cv.destroyAllWindows()

svm_model = load('/Volumes/SSD/ProjetImage/ProjetImage/scripts/SVM_COLOR_AND_TEXTURE.joblib')
scaler = load('/Volumes/SSD/ProjetImage/ProjetImage/scripts/scaler_color_and_texture.joblib')

image_path = '/Volumes/SSD/ProjetImage/valset/128.jpg'
predict_with_combined_features(image_path, svm_model, scaler)
