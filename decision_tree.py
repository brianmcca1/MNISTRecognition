
from sklearn import tree
import numpy as np
import prepare_data

# load data set
(training_images, training_labels,
 validation_images, validation_labels,
 test_images, test_labels) = prepare_data.get_data()

# create classifier and fit to training data
classifier = tree.DecisionTreeClassifier(criterion='entropy')
classifier.fit(training_images, training_labels)

# score with test data
print("Decision Tree Score:", classifier.score(test_images, test_labels))


np.set_printoptions(precision=3)
np.set_printoptions(suppress=True)
np.set_printoptions(threshold=np.nan)
prediction = classifier.predict(validation_images)

# Construct the confusion matrix

num_correct = 0
confusion_matrix = np.zeros(shape=[10, 10])

for i in range(len(prediction)):
    confusion_matrix[prediction[i], validation_labels[i]] += 1
    if prediction[i] == validation_labels[i]:
        num_correct += 1

print("Accuracy: " + str(num_correct / len(prediction)))
print("DECISION TREE CONFUSION MATRIX:")
print(confusion_matrix)


