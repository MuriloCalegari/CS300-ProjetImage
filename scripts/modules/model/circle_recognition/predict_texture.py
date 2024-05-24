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
    "2e": "2_euros"
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

svm_model = load('/Volumes/SSD/ProjetImage/ProjetImage/resources/model/svm_model.joblib')

#test_image_path = '/Volumes/SSD/ProjetImage/ProjetImage/Images/248.jpg'
#predictions = predict_with_hog_features(test_image_path, svm_model)

""" for prediction in predictions:
    print(prediction) """

def test_model(dataset="validation_set"):
    testing_dataset = get_parameter(dataset)
    svm_model = load(get_parameter('svm_model_path'))

    y_true = []
    y_pred = []
    predicted_label_counts = Counter()

    for image in testing_dataset:
        print(f"\nTesting image: {image}")
        image_full_path = f"{get_parameter('image_path')}/{image}"
        annotated_image_path = f"{get_parameter('annotations_path')}/{image.split('.')[0]}.json"

        with open(annotated_image_path, "r") as file:
            shapes = json.loads(file.read())['shapes']
            ground_truth_labels = [shape['label'] for shape in shapes]
            y_true.extend(ground_truth_labels)

            print(f"True labels for {image}: {ground_truth_labels}")

        predicted_labels = predict_with_hog_features(image_full_path, svm_model)
        predicted_label_counts.update(predicted_labels)

        if len(predicted_labels) < len(ground_truth_labels):
            predicted_labels.extend(["none"] * (len(ground_truth_labels) - len(predicted_labels)))
        elif len(predicted_labels) > len(ground_truth_labels):
            ground_truth_labels.extend(["none"] * (len(predicted_labels) - len(ground_truth_labels)))

        y_pred.extend(predicted_labels)

    y_true = [label if isinstance(label, str) else label[0] for label in y_true]
    y_pred = [label if isinstance(label, str) else label[0] for label in y_pred]

    print(f"Unique labels in y_true: {set(y_true)}")
    print(f"Unique labels in y_pred: {set(y_pred)}")

    print("\nClassification Report:")
    print(classification_report(y_true, y_pred))

    print("\nConfusion Matrix:")
    cm = confusion_matrix(y_true, y_pred)
    print(cm)

    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=np.unique(y_true + y_pred))
    disp.plot(cmap=plt.cm.Blues)
    plt.show()

    print("\nTotal number of predicted labels for each class:")
    for label, count in predicted_label_counts.items():
        print(f"{label}: {count}")

    return y_true, y_pred



if __name__ == "__main__":
    y_true, y_pred = test_model(dataset="validation_set")