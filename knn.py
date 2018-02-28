
from sklearn.neighbors import KNeighborsClassifier
import numpy as np
import prepare_data

IMAGE_SIZE = 28

# load data set
(training_images, training_labels,
 validation_images, validation_labels,
 test_images, test_labels) = prepare_data.get_data()

# create and train classifier
classifier = KNeighborsClassifier(n_neighbors=4)
classifier.fit(training_images, training_labels)

# predict for test data
np.set_printoptions(precision=3)
np.set_printoptions(suppress=True)
np.set_printoptions(threshold=np.nan)
prediction = classifier.predict(test_images)

# construct the confusion matrix
num_correct = 0
confusion_matrix = np.zeros(shape=[10, 10])

# Change to true if you want to view incorrectly classified images
SHOW_INCORRECT = False
if SHOW_INCORRECT:
    from matplotlib import pyplot as plt

for i in range(len(prediction)):
    confusion_matrix[prediction[i], test_labels[i]] += 1
    if prediction[i] == test_labels[i]:
        num_correct += 1
    elif SHOW_INCORRECT:
        # show incorrect results
        plt.imshow(np.reshape(test_images[i], [IMAGE_SIZE, IMAGE_SIZE]))
        plt.show()
        print(test_labels[i], "classified as", prediction[i])
        input()

print("Accuracy: " + str(num_correct / len(prediction)))
print("KNN CONFUSION MATRIX:")
print(confusion_matrix)