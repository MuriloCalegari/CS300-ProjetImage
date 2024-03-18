import cv2
import numpy as np

def otsu_thresholding(channel):
    _, thresh = cv2.threshold(channel, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return thresh

def classify_coin(coin_image, ref_1_euro):

    coin_hsv = cv2.cvtColor(coin_image, cv2.COLOR_BGR2HSV)
    ref_1_hsv = cv2.cvtColor(ref_1_euro, cv2.COLOR_BGR2HSV)
    
    coin_hsv_resized = cv2.resize(coin_hsv, (ref_1_hsv.shape[1], ref_1_hsv.shape[0]), interpolation=cv2.INTER_AREA)
    
    coin_thresh = otsu_thresholding(coin_hsv_resized[:, :, 1])
    ref_1_thresh = otsu_thresholding(ref_1_hsv[:, :, 1])
    
    xor_result = cv2.bitwise_xor(coin_thresh, ref_1_thresh)
    
    cv2.imshow("Image originale", coin_image)
    cv2.imshow("Image de referebce", ref_1_euro)
    cv2.imshow("Seuillage de la piece", coin_thresh)
    cv2.imshow("Seuillage de reference", ref_1_thresh)
    cv2.imshow("Resultat XOR", xor_result)
    

    cv2.imwrite("Seuillage_de_la_piece.png", coin_thresh)
    cv2.imwrite("Seuillage_de_reference.png", ref_1_thresh)
    cv2.imwrite("Resultat_XOR.png", xor_result)
    
    mean_xor_value = np.mean(xor_result)
    print("Valeur moyenne du résultat XOR:", mean_xor_value)
    
    color_threshold = 100
    if mean_xor_value < color_threshold:
        return "1 Euro"
    else:
        return "Non classé comme 1 Euro"

coin_image = cv2.imread('/home/yassfkh/Desktop/ProjetImage/ProjetImage/scripts/cache/testing_set/1_euro/212_2_1_euro.jpg')
ref_1_euro = cv2.imread('unE.png')

result = classify_coin(coin_image, ref_1_euro)
print(result)

cv2.waitKey(0)
cv2.destroyAllWindows()
