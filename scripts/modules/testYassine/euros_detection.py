import cv2
import numpy as np
import matplotlib.pyplot as plt

def otsu_thresholding(channel):
    _, thresh = cv2.threshold(channel, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return thresh

def classify_coin(coin_image, ref_1_euro, ref_2_euro):
    coin_hsv = cv2.cvtColor(coin_image, cv2.COLOR_BGR2HSV)
    ref_1_hsv = cv2.cvtColor(ref_1_euro, cv2.COLOR_BGR2HSV)
    ref_2_hsv = cv2.cvtColor(ref_2_euro, cv2.COLOR_BGR2HSV)
    
    ref_2_hsv_resized = cv2.resize(ref_2_hsv, (ref_1_hsv.shape[1], ref_1_hsv.shape[0]), interpolation=cv2.INTER_AREA)
    coin_hsv_resized = cv2.resize(coin_hsv, (ref_1_hsv.shape[1], ref_1_hsv.shape[0]), interpolation=cv2.INTER_AREA)
    
    coin_thresh = otsu_thresholding(coin_hsv_resized[:, :, 1])
    ref_1_thresh = otsu_thresholding(ref_1_hsv[:, :, 1])
    ref_2_thresh = otsu_thresholding(ref_2_hsv_resized[:, :, 1])
    
    # apply XOR
    xor_result_1 = cv2.bitwise_xor(coin_thresh, ref_1_thresh)
    xor_result_2 = cv2.bitwise_xor(coin_thresh, ref_2_thresh)
    
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

    plt.subplot(235)
    plt.imshow(xor_result_1, cmap='gray')
    plt.title('Résultat XOR avec référence 1 Euro')
    
    plt.subplot(236)
    plt.imshow(cv2.cvtColor(ref_2_euro, cv2.COLOR_BGR2RGB))
    plt.title('Image de référence 2 Euro')

    plt.figure(figsize=(18, 9))
    plt.subplot(231)
    plt.imshow(ref_2_thresh, cmap='gray')
    plt.title('Seuillage de référence 2 Euro')
    
    plt.subplot(232)
    plt.imshow(xor_result_2, cmap='gray')
    plt.title('Résultat XOR avec référence 2 Euro')
    
    mean_xor_value_1 = np.mean(xor_result_1)
    mean_xor_value_2 = np.mean(xor_result_2)
    
    color_threshold = 90 
    if mean_xor_value_1 < color_threshold:
        classification = "1 Euro"
    elif mean_xor_value_2 < color_threshold:
        classification = "2 Euro"
    else:
        classification = "Non classé comme 1 Euro ou 2 Euro"
    
    plt.text(0.5, -0.1, f"Classification : {classification}", color='red', fontsize=15, ha='left', transform=plt.gca().transAxes)

    plt.tight_layout()
    plt.show()
    
    print("Valeur moyenne du résultat XOR pour 1 Euro:", mean_xor_value_1)
    print("Valeur moyenne du résultat XOR pour 2 Euro:", mean_xor_value_2)
    
    return classification


coin_image = cv2.imread('/home/yassfkh/Desktop/ProjetImage/ProjetImage/scripts/cache/testing_set/1_euro/58_0_1_euro.jpeg')
ref_1_euro = cv2.imread('unE.png')
ref_2_euro = cv2.imread('deuxE.jpg')

result = classify_coin(coin_image, ref_1_euro, ref_2_euro)
print("Classification :", result)



