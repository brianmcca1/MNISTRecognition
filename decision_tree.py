from sklearn import tree
import numpy as np
import prepare_data

IMAGE_SIZE = 28

# -----------------------------------------------------------
# define feature extraction functions

def average_intensity(image):
    """Returns the average pixel density of the input image"""
    return np.average(image)


def detect_vertical_line(image, line_length=18):
    """Evaluates if there is a vertical line in an image"""
    # first reshape image back into a 2D array
    reshaped = np.reshape(image, [IMAGE_SIZE, IMAGE_SIZE])

    # create a line to convolve with
    line = np.ones(line_length)

    max_value = 0
    for column in reshaped.T:
        max_value = max(max_value, max(np.convolve(column, line)))
    return max_value


def detect_horizontal_line(image, line_length=18):
    """Evaluates if there is a horizontal line in an image"""
    # first reshape image back into a 2D array
    reshaped = np.reshape(image, [IMAGE_SIZE, IMAGE_SIZE])

    # create a line to convolve with
    line = np.ones(line_length)

    max_value = 0
    for row in reshaped:
        max_value = max(max_value, max(np.convolve(row, line)))
    return max_value


def enclosed_space(image, threshold = 100):
    """Calculate the number of pixels enclosed in an area of black"""
    # first reshape image back into a 2D array
    reshaped = np.reshape(image, [IMAGE_SIZE, IMAGE_SIZE])

    num_pixels = 0
    for row in reshaped:
        crossed_boundary = False
        row_num_pixels = 0
        # traverse from the left
        index = 0
        for pixel in row:
            if pixel < threshold:
                if crossed_boundary:
                    row_num_pixels = index
                    break
            else:
                crossed_boundary = True
            index += 1
        # Traverse form teh rght
        crossed_boundary = False
        index = 0
        for pixel in np.flip(row, 0):
            if pixel < threshold:
                if crossed_boundary:
                    row_num_pixels += index
                    break
            else:
                crossed_boundary = True
            index += 1
        # add the number of enclosed pixels
        num_pixels += max(0, row_num_pixels)
    return num_pixels


def average_horizontal_std(image):
    """Average standard deviation of rows"""
    return np.average(image.std(0))


def average_vertical_std(image):
    """Average standard deviation of columns"""
    return np.average(image.T.std(0))


# -----------------------------------------------------------

def extract_features(images):
    extracted_images = []
    for image in images:
        extracted_images.append(np.array([
            average_intensity(image),
            detect_vertical_line(image),
            detect_horizontal_line(image),
            enclosed_space(image),
            average_horizontal_std(image),
            average_vertical_std(image)
        ]))
    return extracted_images


# load data set
(training_images, training_labels,
 validation_images, validation_labels,
 test_images, test_labels) = prepare_data.get_data()


# create classifier and fit to training data
baseline_classifier = tree.DecisionTreeClassifier(criterion='entropy')
tuned_classifier = tree.DecisionTreeClassifier(criterion='entropy', max_depth=9, min_samples_leaf=6)
feature_extracted_classifier = tree.DecisionTreeClassifier(criterion='entropy')

# -----------------------------------------------------------
# Select the classifier to be used (comment out all but one)

classifier = baseline_classifier
# classifier = tuned_classifier
# classifier = feature_extracted_classifier

# -----------------------------------------------------------

# reassign to extracted data if necessary
if classifier == feature_extracted_classifier:
    # extract features
    training_images = extract_features(training_images)
    validation_images = extract_features(validation_images)
    test_images = extract_features(test_images)

classifier.fit(training_images, training_labels)

# score with test data
print("Decision Tree Score:", classifier.score(test_images, test_labels))

np.set_printoptions(precision=3)
np.set_printoptions(suppress=True)
np.set_printoptions(threshold=np.nan)
prediction = classifier.predict(test_images)

# Construct the confusion matrix

num_correct = 0
confusion_matrix = np.zeros(shape=[10, 10])

for i in range(len(prediction)):
    confusion_matrix[prediction[i], test_labels[i]] += 1
    if prediction[i] == test_labels[i]:
        num_correct += 1

print("Accuracy: " + str(num_correct / len(prediction)))
print("DECISION TREE CONFUSION MATRIX:")
print(confusion_matrix)
