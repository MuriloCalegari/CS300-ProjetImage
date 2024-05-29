# Détection et Reconnaissance de Pièces en Euros

Ce projet implémente un système de détection et de reconnaissance de pièces en euros en utilisant des techniques de vision par ordinateur et d'apprentissage automatique. Le projet utilise OpenCV pour la détection des pièces, HoG (Histogram of Oriented Gradients) pour l'extraction de caractéristiques, et un classificateur SVM (Support Vector Machine) pour la reconnaissance des pièces.


## Installation


1. Installez les dépendances :
    ```sh
    pip install -r requirements.txt
    ```

## Utilisation

1. Pour exécuter le script principal, naviguez dans le répertoire `ProjetImage/ProjetImage/scripts` :
    ```sh
    cd ProjetImage/ProjetImage/scripts
    python main.py
    ```

2. Vous serez invité à choisir une option. Voici les options disponibles :
    ```plaintext
    Insert option to continue:
          1. Split datasets
          2. Extract labels from dataset
          3. Test model
          4. Generate squared images from dataset
          5. Test detect circles
          6. (Training) Find Hough Parameters
          7. Run coin detection on one image
          8. Run coin recognition on one image
          9. Test coin detection on testing set
          10. Test coin recognition on testing set
    ```

   Choisissez l'option en entrant le numéro correspondant.

