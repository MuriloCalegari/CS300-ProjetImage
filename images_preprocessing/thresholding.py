import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import os
import cv2

def color_threshold(image, object_type):
    h, w, c = image.shape
    mask = np.zeros((h, w), dtype=bool)

    if object_type == "petitesCentimes":
        lower_color = np.array([100, 0, 0])  # Rouge faible, pas de vert, pas de bleu
        upper_color = np.array([230, 100, 100])  # Rouge fort, un peu de vert, un peu de bleu
    elif object_type == "euros":
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        _, mask = cv2.threshold(gray_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    elif object_type == "grossesCentimes":
        lower_color = np.array([0, 100, 100])  # Jaune faible, un peu de vert, un peu de bleu
        upper_color = np.array([50, 255, 255])  # Jaune fort, beaucoup de vert, beaucoup de bleu
        for i in range(h):
            for j in range(w):
                if all(image[i, j] >= lower_color) and all(image[i, j] <= upper_color):
                    mask[i, j] = True

    if object_type != "euros":
        for i in range(h):
            for j in range(w):
                if all(image[i, j] >= lower_color) and all(image[i, j] <= upper_color):
                    mask[i, j] = True

    return mask

# Charger l'image
current_dir = os.path.dirname(os.path.abspath(__file__))
image_path = os.path.join(current_dir, '..', 'Images', '15.JPG')
image = mpimg.imread(image_path)

# Appliquer le seuillage en fonction du type d'objet
object_type = "euros"  # Modifier le type d'objet ici
segmented_image = color_threshold(image, object_type)

# Afficher les résultats
plt.imshow(segmented_image, cmap='gray')
plt.title(f'Segmentation des {object_type}')
plt.axis('off')
plt.show()
