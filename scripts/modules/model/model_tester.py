import json
import numpy as np
import sys
import cv2 as cv
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

sys.path.append("..")  # Add parent folder to sys.path

from modules.utils import get_parameter, get_parameters
from modules.model.metrics_util import compute_jaccard_index
from modules.model.model_wrapper import detect_coins
from modules.model.model_wrapper import find_coins

label_mapping = {
    "50cts": "50_centimes",
    "20cts": "20_centimes",
    "10cts": "10_centimes",
    "5cts": "5_centimes",
    "2cts": "2_centimes",
    "1cts": "1_centime",
    "1e": "1_euro",
    "2e": "2_euro",
   
}

def clean_label(label):
    """Convert label using predefined mapping."""
    return label_mapping.get(label, label)

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

def was_coin_found_given_truth(ground_truth, circle, original_image_shape, threshold=0.5):
    """
    Given a set of ground truth circles and a detected circle,
    return True if the jaccard index between the detected circle and at least one of the ground truth circles
    is greater than the threshold, else False.

    Args:
        ground_truth (set): A set of tuples (center, radius).
        circle (tuple): A tuple (center, radius).

    Returns:
        truth: Corresponding ground truth circle if the detected circle is found, else None.
    """

    for truth in ground_truth:    
        circle_mask = np.zeros(original_image_shape, np.uint8)
        cv.circle(circle_mask, circle[0], circle[1], (255, 255, 255), -1)

        truth_mask = np.zeros(original_image_shape, np.uint8)
        cv.circle(truth_mask, truth[0], truth[1], (255, 255, 255), -1)

        # f1_score = compute_f1(truth_mask, circle_mask)
        jaccard_index = compute_jaccard_index(truth_mask, circle_mask)
        
        if(jaccard_index > threshold):
            return truth

    return None

def is_the_same_detected_coin(detected_coin, circle, original_image_shape, threshold=0.5):
    """
    Given a detected coin and a ground truth circle, return True if the detected coin is the same as the ground truth circle,
    i.e. the jaccard index between the detected coin and the ground truth circle is greater than the threshold, else False.

    Args:
        detected_coin (dict): A tuple (label, center, radius).
        circle (dict): A tuple (label, center, radius).

    Returns:
        bool: True if the detected coin is the same as the ground truth circle, else False.
    """
    circle_mask = np.zeros(original_image_shape, np.uint8)
    cv.circle(circle_mask, circle[1], circle[2], (255, 255, 255), -1)

    detected_mask = np.zeros(original_image_shape, np.uint8)
    cv.circle(detected_mask, detected_coin[1], detected_coin[2], (255, 255, 255), -1)

    jaccard_index = compute_jaccard_index(circle_mask, detected_mask)

    return jaccard_index > threshold

# Test the find_coins() method by computing the intersection with what we expect from the annotated images
# 1. Load the testing dataset from the annotated JSON files.
def test_find_coins(dataset = "validation_set", parameters = get_parameters()):
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

def find_test_coin_in_predicted_set(predicted_coins, test_coin, original_image_shape, threshold=0.5):
    """
    Given a set of predicted coins and a test coin, return the predicted coin that matches the test coin.

    Args:
        predicted_coins (list): A list of tuples (center, radius).
        test_coin (tuple): A tuple (center, radius).
        original_image_shape (tuple): The shape of the original image.
        threshold (float): The threshold for the Jaccard index.
    """
    for found_coin in predicted_coins:
        found_coin_label = found_coin[0]
        test_coin_label = test_coin[0]
        if(is_the_same_detected_coin(found_coin, test_coin, original_image_shape, threshold)
                and found_coin_label == test_coin_label):
            return found_coin

    return None


def test_model(dataset="testing_set"):
    testing_dataset = get_parameter(dataset)

    micro_average_tp = 0
    micro_average_fp = 0
    micro_average_fn = 0

    macro_average_precision_sum = 0
    macro_average_recall_sum = 0

    all_labels = set()
    y_true = []
    y_pred = []
    false_positives = {}  

    for image in testing_dataset:
        print(f"\nTesting image: {image}")
        detected_coins = detect_coins(image, get_parameters())
        print(f"Detected coins: {detected_coins}")

        true_positives_count = 0
        false_positives_count = 0
        false_negatives_count = 0

        annotated_image_path = f"{get_parameter('annotations_path')}/{image.split('.')[0]}.json"
        image_full_path = f"{get_parameter('image_path')}/{image}"
        with open(annotated_image_path, "r") as file:
            shapes = json.loads(file.read())['shapes']
            found_coins = set()  # A set of tuples (label, center, radius)
            not_found_coins = set()  # Idem

            for shape in shapes:
                center = tuple(shape['points'][0])
                center = tuple(map(int, center))
                label = clean_label(shape['label'])
                all_labels.add(label)

                coin = (label, center, int(euclidean_distance(center, shape['points'][1])))
                image_shape = cv.imread(image_full_path).shape

                correctly_found_and_labeled_coin = find_test_coin_in_predicted_set(detected_coins, coin, image_shape)

                if correctly_found_and_labeled_coin is not None:
                    found_coins.add(correctly_found_and_labeled_coin)
                    detected_coins.remove(correctly_found_and_labeled_coin)  # Each coin should only "be found" once
                    true_positives_count += 1
                    y_true.append(label)
                    y_pred.append(correctly_found_and_labeled_coin[0])
                else:
                    not_found = (label, center, int(euclidean_distance(center, shape['points'][1])))
                    not_found_coins.add(not_found)
                    false_negatives_count += 1
                    y_true.append(label)
                    y_pred.append('missed')

            false_positives_count = len(detected_coins)
            for coin in detected_coins:
                all_labels.add(coin[0])

                min_distance = float('inf')
                closest_coin = None
                for shape in shapes:
                    center = tuple(shape['points'][0])
                    center = tuple(map(int, center))
                    label = clean_label(shape['label'])
                    coin_distance = euclidean_distance(center, coin[1])
                    if coin_distance < min_distance:
                        min_distance = coin_distance
                        closest_coin = (label, center, int(euclidean_distance(center, shape['points'][1])))

                if closest_coin is not None:
                    y_true.append(closest_coin[0])
                    y_pred.append(coin[0])
                else:
                    y_true.append('None')
                    y_pred.append(coin[0])

                if closest_coin is not None:
                    if closest_coin[0] in false_positives:
                        false_positives[closest_coin[0]].append((closest_coin[1:], coin[1:]))  
                    else:
                        false_positives[closest_coin[0]] = [(closest_coin[1:], coin[1:])]  

            # Add to the global metrics
            micro_average_tp += true_positives_count
            micro_average_fp += false_positives_count
            micro_average_fn += false_negatives_count

            precision = true_positives_count / (true_positives_count + false_positives_count) if (true_positives_count + false_positives_count) > 0 else 0
            recall = true_positives_count / (true_positives_count + false_negatives_count) if (true_positives_count + false_negatives_count) > 0 else 0
            f1_score = 0 if (precision == 0 or recall == 0) else 2 * (precision * recall) / (precision + recall)

            macro_average_precision_sum += precision
            macro_average_recall_sum += recall

            print(f"Found coins: {found_coins}")
            print(f"Missed coins: {not_found_coins}")
            print(f"TP: {true_positives_count}, FP: {false_positives_count}, FN: {false_negatives_count}")

    print("\n\nOverall results:")
    print(f"TP: {micro_average_tp}, FP: {micro_average_fp}, FN: {micro_average_fn}")
    precision = micro_average_tp / (micro_average_tp + micro_average_fp) if (micro_average_tp + micro_average_fp) > 0 else 0
    recall = micro_average_tp / (micro_average_tp + micro_average_fn) if (micro_average_tp + micro_average_fn) > 0 else 0
    f1_score = 0 if (precision == 0 or recall == 0) else 2 * (precision * recall) / (precision + recall)
    print("\nMicro-Average results:")
    print(f"Precision: {precision}, Recall: {recall}, F1: {f1_score}")

    # The macro-average is the average precision and recall over all the images
    print("\nMacro-Average results:")
    macro_average_precision = macro_average_precision_sum / len(testing_dataset)
    macro_average_recall = macro_average_recall_sum / len(testing_dataset)
    macro_average_f1 = 0 if (macro_average_precision == 0 or macro_average_recall == 0) else 2 * (macro_average_precision * macro_average_recall) / (macro_average_precision + macro_average_recall)
    print(f"Precision: {macro_average_precision}, Recall: {macro_average_recall}, F1: {macro_average_f1}")

    # generate confusion matrix
    all_labels = sorted(list(all_labels))
    confusion_mat = confusion_matrix(y_true, y_pred, labels=all_labels)
    disp = ConfusionMatrixDisplay(confusion_matrix=confusion_mat, display_labels=all_labels)
    disp.plot(cmap=plt.cm.Blues)
    plt.title('Confusion Matrix')
    plt.show()

    return {
        "precision": precision,
        "recall": recall,
        "f1": f1_score,
        "confusion_matrix": confusion_mat
    }

