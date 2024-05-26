import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from os.path import join
import cv2 as cv
import numpy as np
from joblib import load

from modules.model.circle_detection.hough_transform import detect_circles
from modules.model.circle_detection.hough_transform import extract_hog_features
from modules.model.circle_recognition.predict_texture import pad_features


mapping = {
    "50cts": "50_centimes",
    "20cts": "20_centimes",
    "10cts": "10_centimes",
    "5cts": "5_centimes",
    "2cts": "2_centimes",
    "1cts": "1_centimes",
    "1e": "1_euro",
    "2e": "2_euro",
    "1_euros": "1_euro",
    "2_euros": "2_euro",
    "50_centimes_inverse" : "50_centimes",
    "20_centimes_inverse" : "20_centimes",
    "10_centimes_inverse" : "10_centimes",
    "5_centimes_inverse" : "5_centimes",
    "2_centimes_inverse" : "2_centimes",
    "1_centime_inverse" : "1_centimes",
    "1_euro_inverse" : "1_euro",
    "2_euros_inverse" : "2_euros"

}


def detect_coins(image_file, parameters):
    """
    High level function to detect coins in an image.

    Parameters:
    - image_file (str): The path to the image file.
    - parameters (dict): A dictionary loaded from parameters.json.

    Returns:
    - tuple: A tuple of tuples with the following structure:
        - label (str): The label of the coin.
        - center (tuple): The center coordinates of the coin (x, y).
        - radius (int): The radius of the coin.

    Example:
    (
        ("2_euro", (100, 100), 20),
        ("1_euro", (200, 200), 15),
        ...
    )
    """
    image_path = join(parameters['image_path'], image_file)

    svm_model = load(parameters['svm_model_path'])

    coins = find_coins(image_path, parameters)

    if len(coins) ==  0:
        print("No coins detected.")
        return ()
    
    labeled_coins = label_coins(image_path, coins, svm_model)


    print(f"Labeled coins: {labeled_coins}")
    return list(labeled_coins)

def find_coins(image_path, parameters):
    """
    Find the coins in the image.

    Args:
        image (numpy.ndarray): The input image.
        parameters (dict): A dictionary of parameters for the coin detection algorithm.

    Returns:
        set: A set of tuples (center, radius) representing the detected coins.
    """
    return detect_circles(image_path)

def label_coins(image_path, coins, svm_model):
    image = cv.imread(image_path)
    if image is None:
        print("Image non trouvée.")
        return []

    scale_percent = 50
    width = int(image.shape[1] * scale_percent / 100)
    height = int(image.shape[0] * scale_percent / 100)
    dim = (width, height)
    resized_image = cv.resize(image, dim, interpolation=cv.INTER_AREA)

    labeled_coins = []
    for coin in coins:
        center, radius = coin[:2]
        x, y = center
        # redimensionner les coordonnées que pour le modèle SVM
        x_resized, y_resized, r_resized = int(x * scale_percent / 100), int(y * scale_percent / 100), int(radius * scale_percent / 100)
        crop_img = resized_image[y_resized-r_resized:y_resized+r_resized, x_resized-r_resized:x_resized+r_resized]
        if crop_img.size == 0:
            print("Région croppée vide.")
            continue

        hog_features, hog_image = extract_hog_features(crop_img)
        hog_features_padded = pad_features(hog_features)
        hog_features_padded = np.array(hog_features_padded).reshape(1, -1)
        prediction = svm_model.predict(hog_features_padded)
        label = str(prediction[0])

        label = mapping.get(label, label)

        # on utilise les coordonnées d'origine pour l'étiquetage
        labeled_coins.append((label, (x, y), radius))

        #cv.putText(image, label, (x - radius, y - radius - 10), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    #cv.imshow('Labeled Coins', image)
    #cv.waitKey(0)
    #cv.destroyAllWindows()

    return labeled_coins