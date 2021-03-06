import numpy as np
import io
from ConfusionMatrix import ConfusionMatrix
from ConfidenceHistogram import ConfidenceHistogram
import os

class Analyzer:
    def __init__(self, classes):
        self.matrix = ConfusionMatrix(classes)
        self.histogram = ConfidenceHistogram()
        self.total_predictions = 0
        self.correct_predictions = 0

    def accumulate(self, predictions, labels):
        predicted_indices = np.zeros([len(labels)], dtype='int32')
        for i in range(len(labels)):
            predicted_indices[i] = np.argmax(predictions[i])
            self.total_predictions += 1
            if predicted_indices[i] == labels[i]:
                self.correct_predictions += 1
        self.matrix.accumulate(labels, predicted_indices)
        self.histogram.accumulate(predictions, labels)

    def save(self, path):
        os.makedirs(path, exist_ok=True)
        self.matrix.save_csv(os.path.join(path, 'confusion.csv'))
        self.histogram.save_csv(os.path.join(path, 'confidence.csv'))
        with open(os.path.join(path, 'summary.txt'), 'w') as file:
            file.write('Total predictions: {}\n'.format(self.total_predictions))
            file.write('Correct predictions: {}\n'.format(self.correct_predictions))
            file.write('Correct precentage: {}\n'.format(self.correct_predictions / self.total_predictions))
