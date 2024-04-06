import cv2
import numpy as np
import matplotlib.pyplot as plt

def otsu_thresholding(channel):
    _, thresh = cv2.threshold(channel, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return thresh

def hamming_distance(x, y):
    return np.sum(x != y) / float(x.size)

def classify_coin(coin_image, ref_1_euro, ref_2_euro):
    coin_hsv = cv2.cvtColor(coin_image, cv2.COLOR_BGR2HSV)
    ref_1_hsv = cv2.cvtColor(ref_1_euro, cv2.COLOR_BGR2HSV)
    ref_2_hsv = cv2.cvtColor(ref_2_euro, cv2.COLOR_BGR2HSV)

    ref_2_hsv_resized = cv2.resize(ref_2_hsv, (ref_1_hsv.shape[1], ref_1_hsv.shape[0]), interpolation=cv2.INTER_AREA)
    coin_hsv_resized = cv2.resize(coin_hsv, (ref_1_hsv.shape[1], ref_1_hsv.shape[0]), interpolation=cv2.INTER_AREA)

    coin_thresh = otsu_thresholding(coin_hsv_resized[:, :, 1])
    ref_1_thresh = otsu_thresholding(ref_1_hsv[:, :, 1])
    ref_2_thresh = otsu_thresholding(ref_2_hsv_resized[:, :, 1])

    # Calculer la distance de Hamming entre la pièce et les références
    hamming_dist_1 = hamming_distance(coin_thresh.flatten(), ref_1_thresh.flatten())
    hamming_dist_2 = hamming_distance(coin_thresh.flatten(), ref_2_thresh.flatten())

    plt.figure(figsize=(18, 9))

    plt.subplot(231)
    plt.imshow(cv2.cvtColor(coin_image, cv2.COLOR_BGR2RGB))
    plt.title('Image originale')

    plt.subplot(232)
    plt.imshow(cv2.cvtColor(ref_1_euro, cv2.COLOR_BGR2RGB))
    plt.title('Image de référence 1 Euro')

    plt.subplot(233)
    plt.imshow(coin_thresh, cmap='gray')
    plt.title('Seuillage de la pièce')

    plt.subplot(234)
    plt.imshow(ref_1_thresh, cmap='gray')
    plt.title('Seuillage de référence 1 Euro')

    plt.subplot(236)
    plt.imshow(cv2.cvtColor(ref_2_euro, cv2.COLOR_BGR2RGB))
    plt.title('Image de référence 2 Euro')

    plt.figure(figsize=(18, 9))
    plt.subplot(231)
    plt.imshow(ref_2_thresh, cmap='gray')
    plt.title('Seuillage de référence 2 Euro')



    if hamming_dist_1 < hamming_dist_2:
        classification = "1 Euro"
    elif hamming_dist_2 < hamming_dist_1:
        classification = "2 Euro"
    else:
        classification = "Non classé comme 1 Euro ou 2 Euro"

    plt.text(0.5, -0.1, f"Classification : {classification}", color='red', fontsize=15, ha='left', transform=plt.gca().transAxes)

    plt.tight_layout()
    plt.show()

    print(f"Distance de Hamming pour 1 Euro: {hamming_dist_1:.4f}")
    print(f"Distance de Hamming pour 2 Euro: {hamming_dist_2:.4f}")
    print(f"Classification de la pièce: {classification}")
    
    return classification

coin_image = cv2.imread('/home/yassfkh/Desktop/ProjetImage/ProjetImage/scripts/cache/testing_set/1_euro/68_2_1_euro.jpg')
ref_1_euro = cv2.imread('/home/yassfkh/Desktop/ProjetImage/ProjetImage/scripts/cache/testing_set/1_euro/63_1_1_euro.JPG')
ref_2_euro = cv2.imread('/home/yassfkh/Desktop/ProjetImage/ProjetImage/scripts/cache/testing_set/2_euro/63_2_2_euro.JPG')

result = classify_coin(coin_image, ref_1_euro, ref_2_euro)
print("Classification :", result)

