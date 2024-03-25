import json
import numpy as np
import sys
import cv2 as cv

sys.path.append("..")  # Add parent folder to sys.path

from modules.utils import get_parameter, get_parameters
from modules.model.model_wrapper import detect_coins
from modules.model.model_wrapper import find_coins

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

def was_coin_found_given_truth(ground_truth, circle, original_image_shape, threshold=0.8):
    """
    Given a set of ground truth circles and a detected circle,
    return True if the intersection area between the detected circle and at least one of the ground truth circles
    is greater than the threshold, else False.

    Args:
        ground_truth (set): A set of tuples (center, radius).
        circle (tuple): A tuple (center, radius).

    Returns:
        bool: True if the detected circle is within the radius of any of the ground truth circles, else False.
    """

    for truth in ground_truth:    
        circle_mask = np.zeros(original_image_shape, np.uint8)
        cv.circle(circle_mask, circle[0], circle[1], (255, 255, 255), -1)

        truth_mask = np.zeros(original_image_shape, np.uint8)
        cv.circle(truth_mask, truth[0], truth[1], (255, 255, 255), -1)

        f1_score =compute_f1(truth_mask, circle_mask)
        
        if(f1_score > threshold):
            return truth

    return None

# Test the find_coins() method by computing the intersection with what we expect from the annotated images
# 1. Load the testing dataset from the annotated JSON files. For each circle, 
def test_find_coins(dataset = "testing_set", parameters = get_parameters()):
    testing_data = get_parameter(dataset)

    micro_average_tp = 0
    micro_average_fp = 0
    micro_average_fn = 0

    macro_average_precision_sum = 0
    macro_average_recall_sum = 0

    for image in testing_data:
        print(f"\nTesting image: {image}")
        image_full_path = f"{get_parameter('image_path')}/{image}"
        annotated_image_path = f"{get_parameter('annotations_path')}/{image.split('.')[0]}.json"

        detected_coins = find_coins(image_full_path, parameters)
        true_positives_count = 0
        false_positives_count = 0
        false_negatives_count = 0
        # true_negatives_count = 0

        ground_truth = set()

        with open(annotated_image_path, "r") as file:
            shapes = json.loads(file.read())['shapes']

            image_data = cv.imread(image_full_path)

            ## Create blank image with the same size as the original
            masked_ground_truth = np.zeros((image_data.shape[0], image_data.shape[1], 3), np.uint8)

            for shape in shapes:
                center = tuple(map(int, shape['points'][0]))  # Convert center to integers
                label = shape['label']
                radius = int(euclidean_distance(center, shape['points'][1]))

                cv.circle(masked_ground_truth, center, radius, (255, 255, 255), -1)
                ground_truth.add((center, radius))

                side_by_side = np.hstack((image_data, masked_ground_truth))

        masked_model = np.zeros((image_data.shape[0], image_data.shape[1], 3), np.uint8)

        for coin in detected_coins:
            center = coin[0]
            radius = coin[1]
            cv.circle(masked_model, center, radius, (255, 255, 255), -1)

            found_truth = was_coin_found_given_truth(ground_truth, coin, image_data.shape)

            if(found_truth is not None):
                true_positives_count+=1
                ground_truth.remove(found_truth)
            else:
                false_positives_count+=1

        false_negatives_count = len(ground_truth)

        # Add to the global metrics
        micro_average_tp += true_positives_count
        micro_average_fp += false_positives_count
        micro_average_fn += false_negatives_count

        precision = true_positives_count / (true_positives_count + false_positives_count) if (true_positives_count + false_positives_count) > 0 else 0
        recall = true_positives_count / (true_positives_count + false_negatives_count) if (true_positives_count + false_negatives_count) > 0 else 0
        f1_score = 0 if (precision == 0 or recall == 0) else 2 * (precision * recall) / (precision + recall)

        macro_average_precision_sum += precision
        macro_average_recall_sum += recall

        print(f"TP: {true_positives_count}, FP: {false_positives_count}, FN: {false_negatives_count}")
        print(f"Precision: {precision}, Recall: {recall}, F1: {f1_score}")
        side_by_side = np.hstack((image_data, masked_model))
        # cv.imshow("side_by_side", side_by_side)
        # cv.waitKey(0)

    # Compute global precision, recall and F1 score with the micro average method
    micro_average_precision = micro_average_tp / (micro_average_tp + micro_average_fp) if (micro_average_tp + micro_average_fp) > 0 else 0
    micro_average_recall = micro_average_tp / (micro_average_tp + micro_average_fn)
    
    if(micro_average_precision == 0 or micro_average_recall == 0):
        micro_average_f1 = 0
    else:
        micro_average_f1 = 2 * (micro_average_precision * micro_average_recall) / (micro_average_precision + micro_average_recall)

    print(f"Micro-Average results:")
    print(f"TP: {micro_average_tp}, FP: {micro_average_fp}, FN: {micro_average_fn}")
    print(f"Precision: {micro_average_precision}, Recall: {micro_average_recall}, F1: {micro_average_f1}")

    # Compute global precision, recall and F1 score with the macro average method
    macro_average_precision = macro_average_precision_sum / len(testing_data)
    macro_average_recall = macro_average_recall_sum / len(testing_data)

    if(macro_average_precision == 0 or macro_average_recall == 0):
        macro_average_f1 = 0
    else:
        macro_average_f1 = 2 * (macro_average_precision * macro_average_recall) / (macro_average_precision + macro_average_recall)

    print(f"Macro-Average results:")
    print(f"Precision: {macro_average_precision}, Recall: {macro_average_recall}, F1: {macro_average_f1}")

        # # Compute intersection of masked_ground_truth and masked_model
        # true_positives = cv.bitwise_and(masked_ground_truth, masked_model)
        # false_positives = cv.subtract(masked_model, masked_ground_truth)
        # false_negatives = cv.subtract(masked_ground_truth, masked_model)
        # true_negatives = cv.bitwise_not(cv.bitwise_or(masked_ground_truth, masked_model))

        # # Convert each to gray scale
        # true_positives = cv.cvtColor(true_positives, cv.COLOR_BGR2GRAY)
        # false_positives = cv.cvtColor(false_positives, cv.COLOR_BGR2GRAY)
        # false_negatives = cv.cvtColor(false_negatives, cv.COLOR_BGR2GRAY)
        # true_negatives = cv.cvtColor(true_negatives, cv.COLOR_BGR2GRAY)

        # # Now count the number of pixels which are still 1
        # true_positives_count = cv.countNonZero(true_positives)
        # false_positives_count = cv.countNonZero(false_positives)
        # false_negatives_count = cv.countNonZero(false_negatives)
        # true_negatives_count = cv.countNonZero(true_negatives)

        # total_pixels = image_data.shape[0] * image_data.shape[1]

        # # Compute the accuracy
        # accuracy = (true_positives_count + true_negatives_count) / total_pixels
        # print(f"Accuracy: {accuracy}")

        # cv.imshow("Annotated image", side_by_side)
        # cv.waitKey(0)
    
    return {
        "micro_average": {
            "precision": micro_average_precision,
            "recall": micro_average_recall,
            "f1": micro_average_f1
        },
        "macro_average": {
            "precision": macro_average_precision,
            "recall": macro_average_recall,
            "f1": macro_average_f1
        }
    }

def compute_f1(masked_ground_truth, masked_model):
    """
    Compute the F1 score given the ground truth and the model's output.
    """
    true_positives = cv.bitwise_and(masked_ground_truth, masked_model)
    false_positives = cv.subtract(masked_model, masked_ground_truth)
    false_negatives = cv.subtract(masked_ground_truth, masked_model)
    true_negatives = cv.bitwise_not(cv.bitwise_or(masked_ground_truth, masked_model))

    # Convert each to gray scale
    true_positives = cv.cvtColor(true_positives, cv.COLOR_BGR2GRAY)
    false_positives = cv.cvtColor(false_positives, cv.COLOR_BGR2GRAY)
    false_negatives = cv.cvtColor(false_negatives, cv.COLOR_BGR2GRAY)
    true_negatives = cv.cvtColor(true_negatives, cv.COLOR_BGR2GRAY)

    # Now count the number of pixels which are still 1
    true_positives_count = cv.countNonZero(true_positives)
    false_positives_count = cv.countNonZero(false_positives)
    false_negatives_count = cv.countNonZero(false_negatives)
    # true_negatives_count = cv.countNonZero(true_negatives)

    precision = true_positives_count / (true_positives_count + false_positives_count)
    recall = true_positives_count / (true_positives_count + false_negatives_count)

    if(precision == 0 or recall == 0):
        return 0

    f1 = 2 * (precision * recall) / (precision + recall)

    return f1

def test_model(dataset = "testing_set"):
    testing_dataset = get_parameter(dataset)

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
