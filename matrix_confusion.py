import joblib
from sklearn import metrics
from sklearn.metrics import ConfusionMatrixDisplay
import matplotlib.pyplot as plt
from skimage.feature import hog
import numpy as np


def plot_confusion_matrix_train(y_test, predicted):
      confusion_matrix = metrics.confusion_matrix(y_test, predicted)
      disp = ConfusionMatrixDisplay(
      confusion_matrix, display_labels=np.unique(y_test))
      disp.plot(cmap='viridis', values_format='d')
      plt.title("Confusion Matrix")
      plt.show()

