import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from os.path import join
import cv2 as cv

from modules.model.circle_detection.hough_transform import detect_circles

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

    # Detect coins in the image
    coins = find_coins(image_path, parameters)

    labeled_coins = label_coins(image_path, coins, parameters)

    return labeled_coins

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

def label_coins(image_path, coins, parameters):
    # For each coin just add a dummy label
    return [(f"50_centimes", center, radius) for i, (center, radius) in enumerate(coins)]