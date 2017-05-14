import random
import math
import numpy as np
import io

from Example import Example
from Recognizer import Recognizer
from SQLBase import SQLBase
from DataAugumenter import augument


class MyApp:
    def __init__(self, **kwargs):
        self.should_exit = False
        self.should_load = True
        self.should_save = True
        #self.base = SQLBase('MotionGesture.db')
        self.base = SQLBase('vive.db', 'vive')
        
        self.recognizer = Recognizer(self.base.feature_count, self.base.class_count, self.base.max_length)
        random.seed()

    def do_random_science(self):
        train_set, test_set = self.base.get_large_sets()
        self.train_nn(train_set)
        self.test_nn(test_set)

    def do_user_independent_science(self):
        repeats = 1
        with open('user independent.txt', 'w', 1) as f:
            for i in range(repeats):
                train_set, test_set = self.base.get_user_independent_sets()
                #train_set, test_set = self.base.get_large_sets()
                self.train_nn(train_set)
                percentage = self.test_nn(test_set)
                f.write(str(percentage) + '\n')
                self.recognizer.reset()

    def do_user_dependent_science(self):
        with open('user dependent.txt', 'w', 1) as f:
            for user in self.base.users.keys():
                train_set, test_set = self.base.get_user_dependent_sets(user)
                self.recognizer.reset()
                print('Training with user {}'.format(user))
                self.train_nn(train_set)
                self.test_nn(test_set)

    def train_nn(self, train_set):
        no_improvement_limit = 20
        batch_size = 200
        best_cross_entropy = math.inf
        epochs_without_improvement = 0

        train_examples, train_labels, train_lengths = None, None, None
        
        batched_training = len(train_set) > batch_size
        if not batched_training:
            train_examples, train_labels, train_lengths = Example.to_numpy(train_set)
            
        augumenter = None

        while True:
            # train on given set
            cross_entropy = 0.0
            if batched_training:
                batch_count = 0
                random.shuffle(train_set)
                for i in range(0, len(train_set), batch_size):
                    batch_count += 1
                    train_examples, train_labels, train_lengths = Example.to_numpy(train_set[i:i+batch_size], augumenter)
                    cross_entropy += self.recognizer.train(train_examples, train_labels, train_lengths)
                cross_entropy /= batch_count
            else:
                cross_entropy = self.recognizer.train(train_examples, train_labels, train_lengths)
            
            # test if cross entropy improves
            if cross_entropy < best_cross_entropy:
                best_cross_entropy = cross_entropy
                epochs_without_improvement = 0
                #self.recognizer.save()
                print("Training cross entropy improved to {}".format(cross_entropy))
                if False and augumenter == None and best_cross_entropy < 0.5:
                    print("Enabling augumenter")
                    augumenter = augument
                    best_cross_entropy = math.inf
                #self.test_nn(self.test_set)
            else:
                epochs_without_improvement += 1
                print('.', end='', flush=True)


            # test if training should end
            if epochs_without_improvement >= no_improvement_limit or cross_entropy < 0.01:
                print("Long time without improvement, ending training")
                break

        #self.nn.export_to_protobuffer("./export")

    def test_nn(self, test_set):
        batch_size = 200
        cross_entropy, percentage = (0, 0)

        confusion_matrix = np.zeros([self.base.class_count] * 2, 'int32')

        batch_count = 0
        for i in range(0, len(test_set), batch_size):
            batch_count += 1
            test_examples, test_labels, test_lengths = Example.to_numpy(test_set[i:i+batch_size])
            c, p, predictions, total_predictions = self.recognizer.test(test_examples, test_labels, test_lengths)
            cross_entropy += c
            percentage += p  
            for (actual, prediction) in zip(test_labels, predictions):
                predicted = np.argmax(prediction)
                confusion_matrix[actual, predicted] += 1

        cross_entropy /= batch_count
        percentage /= batch_count
        print("Test cross entropy {}, percentage correct {:.2%}".format(cross_entropy, percentage))
        np.set_printoptions(linewidth=200, formatter={'int' : lambda x: '%3d' % x})
        for (row, name) in zip(confusion_matrix, self.base.gesture_name):
            print(row, end='')
            print(name)
        return percentage

    def save_results(predictions):
        for batch in predictions:
            for row in batch:
                pass



if __name__ == '__main__':
    app = MyApp()
    app.should_load = False
    app.should_save = True
    app.do_user_independent_science()
    #app.do_user_dependent_science()
