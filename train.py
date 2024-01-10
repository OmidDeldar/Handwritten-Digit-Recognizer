import joblib
from sklearn import datasets
from sklearn.model_selection import train_test_split
from skimage.feature import hog
from sklearn.svm import LinearSVC
import numpy as np
from sklearn.metrics import ConfusionMatrixDisplay
import matplotlib.pyplot as plt
from sklearn import metrics
from matrix_confusion import plot_confusion_matrix_train

# Load the MNIST dataset (subset of the Digits dataset.)
mnist = datasets.fetch_openml("mnist_784", as_frame=False, parser='liac-arff')
features = np.array(mnist.data.astype(int))
labels = np.array(mnist.target.astype(int))

list_hog_fd = []
for feature in features:
    fd, _ = hog(feature.reshape((28, 28)), orientations=9, pixels_per_cell=(
        14, 14), cells_per_block=(1, 1), visualize=True)
    list_hog_fd.append(fd)
hog_features = np.array(list_hog_fd, 'float64')

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    hog_features, labels, test_size=0.2, random_state=42)

# Train the Linear SVM with dual set to False
clf = LinearSVC(dual=False)
clf.fit(X_train, y_train)
predicted = clf.predict(X_test)
# Save the classifier
joblib.dump(clf, "digits_cls.pkl", compress=3)

# Compute and display the confusion matrix for the test set
plot_confusion_matrix_train(y_test, predicted)
