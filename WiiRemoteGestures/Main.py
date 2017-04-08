import random
import math
import numpy as np

from Example import Example
from Recognizer import Recognizer
from SQLBase import SQLBase


class MyApp:
    def __init__(self, **kwargs):
        self.should_exit = False
        self.should_load = True
        self.should_save = True
        self.base = SQLBase('MotionGesture.db')
        self.recognizer = Recognizer(self.base.feature_count, self.base.class_count, self.base.max_length)
        random.seed()

    def do_science(self):
        user = random.choice(list(self.base.users.keys()))
        #train_set, self.test_set = self.base.get_user_dependent_sets(user)
        #train_set, self.test_set = self.base.get_large_sets()
        train_set, self.test_set = self.base.get_user_independent_sets()
        self.train_nn(train_set)
        self.test_nn(self.test_set)

    def train_nn(self, train_set):
        no_improvement_limit = 20
        batch_size = 200
        best_cross_entropy = math.inf
        epochs_without_improvement = 0

        train_examples, train_labels, train_lengths = None, None, None
        
        batched_training = len(train_set) > batch_size
        if not batched_training:
            train_examples, train_labels, train_lengths = Example.to_numpy(train_set)
            
        while True:
            # train on given set
            cross_entropy = 0.0
            if batched_training:
                batch_count = 0
                random.shuffle(train_set)
                for i in range(0, len(train_set), batch_size):
                    batch_count += 1
                    train_examples, train_labels, train_lengths = Example.to_numpy(train_set[i:i+batch_size])
                    cross_entropy += self.recognizer.train(train_examples, train_labels, train_lengths)
                cross_entropy /= batch_count
            else:
                cross_entropy = self.recognizer.train(train_examples, train_labels, train_lengths)
            
            # test if cross entropy improves
            if cross_entropy < best_cross_entropy:
                best_cross_entropy = cross_entropy
                epochs_without_improvement = 0
                #self.recognizer.save()
                print("\nTraining cross entropy improved to {}".format(cross_entropy))
                self.test_nn(self.test_set)
            else:
                epochs_without_improvement += 1
                print('.', end='', flush=True)


            # test if training should end
            if epochs_without_improvement >= no_improvement_limit:
                print("Long time without improvement, ending training")
                break

        #self.nn.export_to_protobuffer("./export")

    def test_nn(self, test_set):
        test_examples, test_labels, test_lengths = Example.to_numpy(test_set) 
        cross_entropy, percentage, _, total_pred = self.recognizer.test(test_examples, test_labels, test_lengths)
        print("Test cross entropy {}, percentage correct {:.2%}".format(cross_entropy, percentage))


if __name__ == '__main__':
    app = MyApp()
    app.should_load = False
    app.should_save = True
    app.do_science()
