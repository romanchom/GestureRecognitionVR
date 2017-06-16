import io

from ViveBase import ViveBase
from WiiBase import WiiBase
from Supervisor import Supervisor
from Analyzer import Analyzer
from Recognizer import Recognizer

class MyApp:
    def __init__(self, **kwargs):
        self.supervisor = Supervisor()
        self.supervisor.recognizer = Recognizer()

    def run_experiments(self):
        wii_name = 'MotionGesture.db'
        vive_name = 'vive.db'
        scenarios = [
            (WiiBase(wii_name, WiiBase.feature_set_position), 'wii_P'),
            (WiiBase(wii_name, WiiBase.feature_set_position_orientation), 'wii_PO'),
            (WiiBase(wii_name, WiiBase.feature_set_acceleration), 'wii_A'),
            (WiiBase(wii_name, WiiBase.feature_set_acceleration_angular_velocity), 'wii_AW'),
            (WiiBase(wii_name, WiiBase.feature_set_full), 'wii_POAW'),
            (ViveBase(vive_name, ViveBase.feature_set_position), 'vive_P'),
            (ViveBase(vive_name, ViveBase.feature_set_position_orientation), 'vive_PO'),
            (ViveBase(vive_name, ViveBase.feature_set_velocity), 'vive_V'),
            (ViveBase(vive_name, ViveBase.feature_set_velocity_angular_velocity), 'vive_VW'),
            (ViveBase(vive_name, ViveBase.feature_set_full), 'vive_POVW'),
        ]

        for (base, name) in scenarios:
            self.supervisor.recognizer.initialize(base.feature_count, base.class_count, base.max_length)
            self.run_user_dependent(name, base)
            self.run_user_independent(name, base)
            #self.run_cross_validate_user_independent(name)


    def run_user_independent(self, name, base):
        analyzer = Analyzer(base.classes)

        sets = base.get_user_independent_sets(6)
        for train_set, test_set in sets:
            self.supervisor.recognizer.reset()
            self.supervisor.train_nn(train_set)
            self.supervisor.test_nn(test_set, analyzer)

        analyzer.save(name + '/user_independent')                

    def run_cross_validate_user_independent(self, name, base):
        analyzer = Analyzer(base.classes)

        mean_percentage = 0 
        all_sets = base.get_cross_validation_sets()
        for i in range(3):
            for sets in all_sets:
                train_set, test_set = sets
                self.supervisor.recognizer.reset()
                self.supervisor.train_nn(train_set)
                self.supervisor.test_nn(test_set, analyzer)

        analyzer.save(name + '/user_independend_cross')

    def run_user_dependent(self, name, base):
        analyzer = Analyzer(base.classes)

        for i in range(3):
            for sets in base.get_user_dependent_sets():
                train_set, test_set, tester = sets
                print('Training with user {}'.format(tester))
                print('Train set: {}, Test set: {}'.format(len(train_set), len(test_set)))
                self.supervisor.recognizer.reset()
                self.supervisor.train_nn(train_set)
                self.supervisor.test_nn(test_set, analyzer)

        analyzer.save(name + '/user_dependent')

    def run_off_hand(self, name, base):
        analyzer = Analyzer(base.classes)

        train_set, test_set = base.get_other_hand_sets()
        self.supervisor.recognizer.reset()
        self.supervisor.train_nn(train_set)
        self.supervisor.test_nn(test_set, analyzer)

        analyzer.save(name + '/off_hand')

if __name__ == '__main__':
    app = MyApp()
    app.run_experiments()
