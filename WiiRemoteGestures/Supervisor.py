import random
import math
import numpy as np

from Recognizer import Recognizer
from Gesture import Gesture

class Supervisor:
    def __init__(self):
        self.no_improvement_limit = 20
        self.batch_size = 400
        self.min_cross_entropy = 0.1

        self.recognizer = Recognizer()
        
        random.seed()


    def train_nn(self, train_set):
        best_cross_entropy = math.inf
        epochs_without_improvement = 0
        
        batch_count = len(train_set) // self.batch_size + 1
        if batch_count > 1:
            batch_size = len(train_set) // batch_count + 1
        else:
            train_examples, train_labels, train_lengths = Gesture.to_numpy(train_set)

        while True:
            # train on given set
            cross_entropy = 0.0
            if batch_count > 1:
                random.shuffle(train_set)
                for i in range(0, len(train_set), batch_size):
                    train_examples, train_labels, train_lengths = Gesture.to_numpy(train_set[i:i+batch_size])
                    cross_entropy += self.recognizer.train(train_examples, train_labels, train_lengths)
                cross_entropy /= batch_count
            else:
                cross_entropy = self.recognizer.train(train_examples, train_labels, train_lengths)
            
            # test if cross entropy improves
            if cross_entropy < best_cross_entropy:
                best_cross_entropy = cross_entropy
                epochs_without_improvement = 0
                print("Training cross entropy improved to {}".format(cross_entropy))
            else:
                epochs_without_improvement += 1
                print('.', end='', flush=True)

            # test if training should end
            if cross_entropy < 0.2 and (epochs_without_improvement >= self.no_improvement_limit or cross_entropy < self.min_cross_entropy):
                print("Long time without improvement, ending training")
                break

        #self.nn.export_to_protobuffer("./export")

    def test_nn(self, test_set, analyzer):
        cross_entropy, percentage = (0, 0)

        data_point_count = len(test_set)
        batch_count = data_point_count // self.batch_size + 1
        batch_size = data_point_count // batch_count + 1

        for i in range(0, data_point_count, batch_size):
            test_examples, test_labels, test_lengths = Gesture.to_numpy(test_set[i:i+batch_size])
            c, p, predictions, total_predictions = self.recognizer.test(test_examples, test_labels, test_lengths)
            count = len(test_examples)
            cross_entropy += c * count
            percentage += p * count
            analyzer.accumulate(predictions, test_labels)

        cross_entropy /= data_point_count
        percentage /= data_point_count
        print("Test cross entropy {}, percentage correct {:.2%}".format(cross_entropy, percentage))
        return percentage