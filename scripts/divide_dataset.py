import os
import random

train_percent = 0.8
val_percent = 0.1
test_percent = 0.1

image_folder = '/home/yassfkh/Desktop/ProjetImage/ProjetImage/Images'
image_filenames = os.listdir(image_folder)

random.shuffle(image_filenames)

num_images = len(image_filenames)
num_train = int(num_images * train_percent)
num_val = int(num_images * val_percent)
num_test = num_images - num_train - num_val

train_filenames = image_filenames[:num_train]
val_filenames = image_filenames[num_train:num_train+num_val]
test_filenames = image_filenames[num_train+num_val:]

train_folder = '/home/yassfkh/Desktop/ProjetImage/DividedDataset/trainset'
val_folder = '/home/yassfkh/Desktop/ProjetImage/DividedDataset/valset'
test_folder = '/home/yassfkh/Desktop/ProjetImage/DividedDataset/testset'
os.makedirs(train_folder, exist_ok=True)
os.makedirs(val_folder, exist_ok=True)
os.makedirs(test_folder, exist_ok=True)

for filename in train_filenames:
    src_path = os.path.join(image_folder, filename)
    dst_path = os.path.join(train_folder, filename)
    os.rename(src_path, dst_path)

for filename in val_filenames:
    src_path = os.path.join(image_folder, filename)
    dst_path = os.path.join(val_folder, filename)
    os.rename(src_path, dst_path)

for filename in test_filenames:
    src_path = os.path.join(image_folder, filename)
    dst_path = os.path.join(test_folder, filename)
    os.rename(src_path, dst_path)
