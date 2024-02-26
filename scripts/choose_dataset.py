from os import listdir
from os.path import isfile, join
from random import shuffle
import math

IMAGE_PATH = "../Images"
propotions = (60, 20, 20) # Training, validation, testing, in percentage

images = shuffle([f for f in listdir(IMAGE_PATH) if isfile(join(IMAGE_PATH, f))])

propotions = (math.floor(propotions[0] / 100 * len(images)), math.floor(propotions[1] / 100 * len(images)), 0)
propotions = (propotions[0], propotions[1], len(images) - propotions[0] - propotions[1])

training_set = images[0:(propotions[0])]
validation_set = images[propotions[0]:(propotions[0] + propotions[1])]
testing_set = images[(propotions[0] + propotions[1]):]

print(training_set)
print(validation_set)
print(testing_set)