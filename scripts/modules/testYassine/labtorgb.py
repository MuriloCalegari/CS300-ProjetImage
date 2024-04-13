import cv2
import numpy as np

# Valeurs L*a*b* en float32
lab = np.array([64.17, 139.69, 140.86], dtype=np.float32)

# Normaliser les valeurs L*a*b* en float32
lab_norm = lab / np.array([100.0, 128.0, 128.0], dtype=np.float32)

# Convertir L*a*b* en RVB
rgb = cv2.cvtColor(lab_norm.reshape(1, 1, -1), cv2.COLOR_LAB2BGR)

# Convertir en uint8 et afficher la couleur
rgb = (rgb * 255).astype(np.uint8)
print("Couleur moyenne RVB : ", rgb.reshape(-1))
