from keras.models import Sequential
from keras.layers import Dense, Activation
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
images = np.load("data/images.npy")
labels = np.load("data/labels.npy")

reshapedImages = []
reshapedLabels = []

# Reshape each image into a flat vector
for image in images:
	reshapedImages.append(np.reshape(image, 784))

# Reshape each label into "one-hot vectors"
for label in labels:
	reshapedLabels.append(utils.to_categorical(label, num_classes=10));


# Separate the images and labels into 10 classes, representing the numbers 0-9
imageClasses = [[0 for x in range(10)] for y in range(79)]

# Note: if the above doesn't work, look into new ways to sort by class


for index, label in enumerate(reshapedLabels):
	i = 0
	while i < 10:
		if label[i] == 1:
			numberImage = NumberImage(reshapedImages[index], label);
			imageClasses[i].append(numberImage)
		i += 1

trainingSet = []
validationSet = []
testSet = []

# Distribute the data into the 3 sets with the following distribution:
# 60% in the Training Set
# 15% in the Validation Set
# 25% in the Test set
# This is done by shuffling each set of data, then iterating through and
# adding the appropriate portion to each set.
i = 0
while i < 10:
	shuffle(imageClasses[i])
	for index, numberImage in enumerate(imageClasses[i]):
		if numberImage != 0:
			if index < (0.6 * len(imageClasses[i])):
				trainingSet.append(numberImage)
			elif index < (0.75 * len(imageClasses[i])):
				validationSet.append(numberImage)
			else:
				testSet.append(numberImage)
	i += 1


# Now that everything is organized, separate into separate objects
trainingSetImages = []
trainingSetLabels = []

validationSetImages = []
validationSetLabels = []

testSetImages = []
testSetLabels = []

for numberImage in trainingSet:
	trainingSetImages.append(numberImage.image)
	trainingSetLabels.append(numberImage.label)
for numberImage in validationSet:
	validationSetImages.append(numberImage.image)
	validationSetLabels.append(numberImage.label)
for numberImage in testSet:
	testSetImages.append(numberImage.image)
	testSetLabels.append(numberImage.label)
# Model Template

model = Sequential() # declare model
model.add(Dense(784, input_shape=(28*28, ), kernel_initializer='lecun_uniform')) # first layer
model.add(Activation('tanh'))
#
#
#
# Fill in Model Here
#
#

model.add(Dense(512, kernel_initializer='lecun_uniform'))
model.add(Activation('tanh'))
model.add(Dense(256, kernel_initializer='lecun_uniform'))
model.add(Activation('tanh'))
model.add(Dense(128, kernel_initializer='lecun_uniform'))
model.add(Activation('tanh'))

model.add(Dense(64, kernel_initializer='lecun_uniform'))
model.add(Activation('tanh'))

model.add(Dense(32, kernel_initializer='lecun_uniform'))
model.add(Activation('tanh'))

model.add(Dense(10, kernel_initializer='he_normal')) # last layer
model.add(Activation('softmax'))


# Compile Model
model.compile(optimizer='sgd',
              loss='categorical_crossentropy', 
              metrics=['accuracy'])

# Train Model
history = model.fit(np.array(trainingSetImages), np.array(trainingSetLabels), 
                    validation_data = (np.array(validationSetImages), np.array(validationSetLabels)), 
                    epochs=100, 
                    batch_size=128)


# Report Results

print(history.history)

np.set_printoptions(precision=3)
np.set_printoptions(suppress=True)
np.set_printoptions(threshold=np.nan)
prediction = model.predict(np.array(testSetImages), batch_size=512)

# Construct the confusion matrix
index = 0;
correctCount = 0;
confusionMatrix = np.zeros(shape=[10, 10])
for row in prediction:
	max = 0;
	maxIndex = 0;
	columnIndex = 0;
	for val in row:
		if val > max:
			max = val
			maxIndex = columnIndex
		columnIndex += 1
	realVal = testSetLabels[index]
	confusionMatrix[maxIndex, getIntFromLabelArray(realVal)] += 1
	if maxIndex == getIntFromLabelArray(realVal):
		correctCount += 1
	index += 1
print("Accuracy: " + str(correctCount / index))
print("ANN CONFUSION MATRIX:")
print(confusionMatrix)




