from os import listdir
from os.path import isfile, join
from random import shuffle
import math
import json
import sys

sys.path.append("..")  # Add parent folder to sys.path

import modules.utils as utils


# Given all of the files in the image_pat,
# return the (training_set, validation_set, testing_set)
def split_datasets(image_path):
    propotions = (60, 20, 20) # Training, validation, testing, in percentage

    images = [f for f in listdir(image_path)
              if (isfile(join(image_path, f)) and ("DS_Store" not in f))]
    shuffle(images)

    propotions = (math.floor(propotions[0] / 100 * len(images)), math.floor(propotions[1] / 100 * len(images)), 0)
    propotions = (propotions[0], propotions[1], len(images) - propotions[0] - propotions[1])

    training_set = images[0:(propotions[0])]
    validation_set = images[propotions[0]:(propotions[0] + propotions[1])]
    testing_set = images[(propotions[0] + propotions[1]):]

    return (training_set, validation_set, testing_set)

def split_and_persist_datasets(data_sets):
    training_set, validation_set, testing_set = split_datasets(data_sets)

    with open("./parameters.json", "r") as file:
        parameters = json.loads(file.read())
        # Apend all three sets to the file
        parameters["training_set"] = training_set
        parameters["validation_set"] = validation_set
        parameters["testing_set"] = testing_set

    with open("./parameters.json", "w") as file:
        json.dump(parameters, file, indent = 4)

def get_labels():
    annotations_path = utils.get_parameter("annotations_path")

    labels = set()

    for f in listdir(annotations_path):
        file_path = join(annotations_path, f)
        if not isfile(join(annotations_path, f)) or not f.endswith(".json"):
            continue

        with open(file_path, "r") as file:
            shapes = json.loads(file.read())["shapes"]
            for shape in shapes:
                labels.add(shape["label"])

    return labels
