from joblib import load
import numpy as np
import cv2 as cv


from modules.model.circle_detection.hough_transform import detect_circles, extract_hog_features, detect_cicles_opencv

def predict_with_color_features(image_path, svm_model, scaler):
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

            lab_image = cv.cvtColor(crop_img, cv.COLOR_BGR2LAB)
            l, a, b = cv.split(lab_image)
            color_features = [np.mean(l), np.mean(a), np.mean(b)]
            
            diameter = 2 * r  
            features = [diameter] + color_features 
            
            features = scaler.transform([features])

            prediction = svm_model.predict(features)
            print("Prédiction pour la pièce détectée :", prediction)

            text = str(prediction[0])
            cv.putText(resized_image, text, (x, y), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv.LINE_AA)

    cv.imshow('Detected Coins with Predictions', resized_image)
    cv.waitKey(0)
    cv.destroyAllWindows()

svm_model = load('/Volumes/SSD/ProjetImage/ProjetImage/scripts/SVM_COLOR.joblib')
scaler = load('scaler.joblib')

image_path = '/Volumes/SSD/ProjetImage/valset/128.jpg'

predict_with_color_features(image_path, svm_model, scaler)
