from keras import utils
from random import shuffle
import numpy as np


# Class to represent the pairing of an image and the corresponding label
class NumberImage:
    def __init__(self, image, label):
        self.image = image
        self.label = label


def getIntFromLabelArray(labelArray):
    index = 0
    for label in labelArray:
        if label == 1:
            return index
        index += 1
    return -1


# PRE-PROCESSING

# Load the .npy files using numpy
images = np.load('data/images.npy')
labels = np.load('data/labels.npy')

reshaped_images = []
reshaped_labels = []

# Reshape each image into a flat vector
for image in images:
    reshaped_images.append(np.reshape(image, 784))

# Reshape each label into "one-hot vectors"
for label in labels:
    reshaped_labels.append(utils.to_categorical(label, num_classes=10))

# Separate the images and labels into 10 classes, representing the numbers 0-9
image_classes = [[0 for x in range(10)] for y in range(79)]

# Note: if the above doesn't work, look into new ways to sort by class


for index, label in enumerate(reshaped_labels):
    i = 0
    while i < 10:
        if label[i] == 1:
            number_image = NumberImage(reshaped_images[index], i)
            image_classes[i].append(number_image)
        i += 1


def get_data():
    training_set = []
    validation_set = []
    test_set = []
    # Distribute the data into the 3 sets with the following distribution:
    # 60% in the Training Set
    # 15% in the Validation Set
    # 25% in the Test set
    # This is done by shuffling each set of data, then iterating through and
    # adding the appropriate portion to each set.
    i = 0
    while i < 10:
        shuffle(image_classes[i])
        for index, number_image in enumerate(image_classes[i]):
            if number_image != 0:
                if index < (0.6 * len(image_classes[i])):
                    training_set.append(number_image)
                elif index < (0.75 * len(image_classes[i])):
                    validation_set.append(number_image)
                else:
                    test_set.append(number_image)
        i += 1

    # Now that everything is organized, separate into separate objects
    training_set_images = []
    training_set_labels = []

    validation_set_images = []
    validation_set_labels = []

    test_set_images = []
    test_set_labels = []

    for number_image in training_set:
        training_set_images.append(number_image.image)
        training_set_labels.append(number_image.label)
    for number_image in validation_set:
        validation_set_images.append(number_image.image)
        validation_set_labels.append(number_image.label)
    for number_image in test_set:
        test_set_images.append(number_image.image)
        test_set_labels.append(number_image.label)

    return training_set_images, training_set_labels, \
           validation_set_images, validation_set_labels, \
           test_set_images, test_set_labels
