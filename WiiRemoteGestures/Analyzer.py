import numpy as np
from ConfusionMatrix import ConfusionMatrix
from ConfidenceHistogram import ConfidenceHistogram

class Analyzer:
    def __init__(self, classes):
        self.matrix = ConfusionMatrix(classes)
        self.histogram = ConfidenceHistogram()

    def accumulate(self, predictions, labels):
        predicted_indices = np.zeros([len(labels)], dtype='int32')
        for i in range(len(labels)):
            predicted_indices[i] = np.argmax(predictions[i])
        self.matrix.accumulate(labels, predicted_indices)
        self.histogram.accumulate(predictions, labels)