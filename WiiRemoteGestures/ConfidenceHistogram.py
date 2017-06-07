import numpy as np
import io
import csv


class ConfidenceHistogram:
    def __init__(self, buckets = 20):
        self.buckets = buckets
        self.histogram = np.zeros([buckets], 'int32')


    def accumulate(self, confidences, labels):
        for (confidences, actual) in zip(confidences, labels):
            index = int(confidences[actual] * self.buckets)
            self.histogram[index] += 1

    def save_csv(self, file_name):
        with open(file_name, 'w', newline='') as file:
            writer = csv.writer(file, delimiter=';')
            for row in zip(np.linspace(0, 1, self.buckets, endpoint=False), self.histogram):
                writer.writerow(row)
