import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from os.path import join
import cv2 as cv

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
    # Load all images using matplotlib
    image = mpimg.imread(join(parameters['image_path'], image_file))

    # Detect coins in the image
    coins = find_coins(image, parameters)

    labeled_coins = label_coins(coins, parameters)

    return set()

def find_coins(image, parameters):
    print("Apply Otus thresholding")
    gray_image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)  # Convert image to grayscale
    ret2, th2 = cv.threshold(gray_image, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)
    plt.imshow(th2, cmap="gray", lvmin=0, vmax=255)
    plt.show()
    return set()

def label_coins(coins, parameters):
    return set()