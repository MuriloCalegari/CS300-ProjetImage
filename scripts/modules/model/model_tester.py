import sys

sys.path.append("..")  # Add parent folder to sys.path

from modules.utils import get_parameter, get_parameters
from modules.model.model_wrapper import detect_coins
import json

def squared_distance(p1, p2):
    return (p1[0] - p2[0])**2 + (p1[1] - p2[1])**2

def euclidean_distance(p1, p2):
    return squared_distance(p1, p2)**0.5

def detected_coins_contains(detected_coins, label, center):
    """
    Check if any of the detected coins contains a specific label and center,
    i.e. the labels should match and the expected center should be within the radius of the detected coin.

    Args:
        detected_coins (list): A list of dictionaries representing the detected coins from the model.
        label (str): The label to search for.
        center (tuple): The center coordinates to compare with.

    Returns:
        tuple: A tuple of the form (label, center, radius) if the coin is found, else None.
    """
    for coin in detected_coins:
        if (coin['label'] == label and
            (squared_distance(coin['center'], center) < coin['radius']**2)):
            return coin
    return None

def test_model():
    testing_dataset = get_parameter("testing_set")

    for image in testing_dataset:
        print(f"Testing image: {image}")
        detected_coins = detect_coins(image, get_parameters())

        annotated_image_path = f"{get_parameter('annotations_path')}/{image.split('.')[0]}.json"
        with open(annotated_image_path, "r") as file:
            shapes = json.loads(file.read())['shapes']
            found_coins = set() # A set of tuples (label, center, radius)
            not_found_coins = set() # Idem

            for shape in shapes:
                center = tuple(shape['points'][0])
                label = shape['label']

                found_coin = detected_coins_contains(detected_coins, label, center)

                if found_coin is not None:
                    found_coins.add(found_coin)
                    detected_coins.remove(found_coin) # Each coin should only "be found" once
                else:
                    not_found = (label, center, euclidean_distance(center, shape['points'][1]))
                    not_found_coins.add(not_found)
            print(f"Found coins: {found_coins}")
            print(f"Missed coins: {not_found_coins}")
