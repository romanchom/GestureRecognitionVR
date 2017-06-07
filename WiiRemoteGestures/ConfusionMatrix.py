import numpy as np
import csv
import io

class ConfusionMatrix:
    def __init__(self, classes):
        self.matrix = np.zeros([len(classes)] * 2, 'int32')
        self.classes = classes

    def __str__(self):
        np.set_printoptions(linewidth=200, formatter={'int' : lambda x: '%3d' % x})
        for (row, label) in zip(self.matrix, self.classes):
            print(row, end='')
            print(label)
            
    def save_csv(self, file_name):
        with open(file_name, 'w', newline='') as file:
            writer = csv.writer(file, delimiter=';')
            header = [''] + self.classes + ['']
            writer.writerow(header)
            for (row, label) in zip(self.matrix, self.classes):
                writer.writerow([label] + list(row) + [label])
            writer.writerow(header)

    def accumulate(self, actual, predicted):
        for (a, p) in zip(actual, predicted):
            self.matrix[a, p] += 1

