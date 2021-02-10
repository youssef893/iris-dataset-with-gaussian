from random import randrange as rand
import numpy as np
from sklearn import datasets
import seaborn as sns
from sklearn.metrics import confusion_matrix
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from yellowbrick.contrib.classifier import DecisionViz
import matplotlib.pyplot as plt
# Sizes of train, validate and test datasets

# Load iris data set
iris_dataset = datasets.load_iris()


def calculate_length(validatRatio, testRatio, trainRatio):
    VALIDATE_DATASET_SIZE = len(iris_dataset.target) * validatRatio
    TEST_DATASET_SIZE = len(iris_dataset.target) * testRatio
    TRAIN_DATASET_SIZE = len(iris_dataset.target) * trainRatio

    return VALIDATE_DATASET_SIZE, TEST_DATASET_SIZE, TRAIN_DATASET_SIZE


VALIDATE_DATASET_SIZE, TEST_DATASET_SIZE, TRAIN_DATASET_SIZE = calculate_length(0.3, 0.3, 0.4)

print(len(iris_dataset.target))
print(VALIDATE_DATASET_SIZE)
print(TEST_DATASET_SIZE)
print(TRAIN_DATASET_SIZE)
# Declare Lists
data_validating_dataset = list()
data_training_dataset = list()
data_testing_dataset = list()

label_validating_dataset = list()
label_training_dataset = list()
label_testing_dataset = list()


def draw_boundaries():
    data = datasets.load_iris().data[:, :2]  # we only take the first two features.
    label = np.array(datasets.load_iris().target, dtype=int)  # Take classes names

    data = StandardScaler().fit_transform(data)  # Rescale data

    viz = DecisionViz(
        KNeighborsClassifier(1), title="Nearest Neighbors",
        features=['Sepal Length', 'Sepal Width'], classes=['A', 'B', 'C']  # Determine dimension and no. of classes
    )
    viz.fit(data, label)  # Train data to draw
    viz.draw(data, label)  # Draw Data
    viz.show()


def split_dataset():
    global label_training_dataset, data_training_dataset
    condition = True
    data = list(iris_dataset.data[0:])  # Take copy from data in iris dataset bec iris_dataset.data[0:] do not has pop
    label = list(iris_dataset.target)  # Take copy from labels in iris dataset

    while condition:

        # split dataset into three types
        if len(data_validating_dataset) < VALIDATE_DATASET_SIZE:
            index = rand(0, len(data), 1)
            data_validating_dataset.append(data.pop(index))
            label_validating_dataset.append(label.pop(index))

        elif len(data_testing_dataset) < TEST_DATASET_SIZE:
            index = rand(0, len(data), 1)
            data_testing_dataset.append(data.pop(index))
            label_testing_dataset.append(label.pop(index))

        else:
            data_training_dataset = data
            label_training_dataset = label
            condition = False
            iris_dataset.clear()

    return data_training_dataset, label_training_dataset, data_validating_dataset, \
           label_validating_dataset, data_testing_dataset, label_testing_dataset


# calculate accuracy of the model on test dataset
def calculate_accuracy(label_dataset, prediction_dataset):
    correct = 0
    for i in range(len(label_dataset)):
        if label_dataset[i] == prediction_dataset[i]:
            correct += 1

    return (correct / len(label_dataset)) * 100.0


# Call functions
draw_boundaries()

data_train_dataset, label_train_dataset, data_validate_dataset, \
label_validate_dataset, data_test_dataset, label_test_dataset = split_dataset()

# create gaussian classifier
gaussian = GaussianNB()

# Train iris model using training set
gaussian.fit(data_train_dataset, label_train_dataset)

# gaussian.fit(data_validating_dataset, label_validating_dataset)

# predict the result of test dataset
label_prediction = gaussian.predict(data_test_dataset)

print(calculate_accuracy(label_test_dataset, label_prediction))