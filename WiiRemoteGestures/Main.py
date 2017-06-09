import io

from ViveBase import ViveBase
from Supervisor import Supervisor
from Analyzer import Analyzer

class MyApp:
    def __init__(self, **kwargs):
        self.base = ViveBase()
        self.base.load('vive.db')
        self.supervisor = Supervisor(self.base.feature_count, self.base.class_count, self.base.max_length)

    def do_user_independent_science(self):
        analyzer = Analyzer(self.base.classes)

        repeats = 1
        with open('user independent.txt', 'w', 1) as f:
            for i in range(repeats):
                train_set, test_set = self.base.get_user_independent_sets()
                self.supervisor.recognizer.reset()
                self.supervisor.train_nn(train_set)
                percentage = self.supervisor.test_nn(test_set, analyzer)
                f.write(str(percentage) + '\n')

        analyzer.save('ui')                

    def cross_validate_user_independent(self):
        analyzer = Analyzer(self.base.classes)

        mean_percentage = 0 
        all_sets = self.base.get_cross_validation_sets()
        with open('user independent cv.txt', 'w', 1) as f:
            for i in range(3):
                for sets in all_sets:
                    train_set, test_set = sets
                    self.supervisor.recognizer.reset()
                    self.supervisor.train_nn(train_set)
                    percentage = self.supervisor.test_nn(test_set, analyzer)
                    mean_percentage += percentage
                    f.write("{}\n".format(percentage))

                mean_percentage /= len(all_sets)
                msg = "mean {}\n".format(mean_percentage)
                f.write(msg)
                print(msg)

        analyzer.save('uic')

    def do_user_dependent_science(self):
        analyzer = Analyzer(self.base.classes)

        with open('user dependent.txt', 'w', 1) as f:
            for i in range(3):
                for sets in self.base.get_user_dependent_sets():
                    train_set, test_set, tester = sets
                    print('Training with user {}'.format(tester))
                    print('Train set: {}, Test set: {}'.format(len(train_set), len(test_set)))
                    self.supervisor.recognizer.reset()
                    self.supervisor.train_nn(train_set)
                    self.supervisor.test_nn(test_set, analyzer)

        analyzer.save('ud')

    def do_off_hand_science(self):
        analyzer = Analyzer(self.base.classes)

        train_set, test_set = self.base.get_other_hand_sets()
        self.supervisor.recognizer.reset()
        self.supervisor.train_nn(train_set)
        percentage = self.supervisor.test_nn(test_set, analyzer)
        f.write(str(percentage) + '\n')


if __name__ == '__main__':
    app = MyApp()
    app.should_load = False
    app.should_save = True
    #app.cross_validate_user_independent()
    #app.do_user_dependent_science()
    app.do_off_hand_science()
