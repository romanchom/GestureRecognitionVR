import io

from ViveBase import ViveBase
from WiiBase import WiiBase
from Supervisor import Supervisor
from Analyzer import Analyzer
from Recognizer import Recognizer


wii_name = 'MotionGesture.db'
vive_name = 'vive.db'

class MyApp:
    def __init__(self, **kwargs):
        self.supervisor = Supervisor()
        self.supervisor.recognizer = Recognizer()

    def experiment_1(self):
        # direct comparison to HMM
        scenarios = [
            (WiiBase(wii_name, WiiBase.feature_set_PV), '1/wii_PV'),
            (WiiBase(wii_name, WiiBase.feature_set_AW), '1/wii_AW'),
            (WiiBase(wii_name, WiiBase.feature_set_AWO), '1/wii_AWO'),
            (WiiBase(wii_name, WiiBase.feature_set_PVO), '1/wii_PVO'),
            (WiiBase(wii_name, WiiBase.feature_set_PVOW), '1/wii_PVOW'),
            (WiiBase(wii_name, WiiBase.feature_set_PVOWA), '1/wii_PVOWA'),
            (ViveBase(vive_name, ViveBase.feature_set_PV), 'vive_PV'),
            (ViveBase(vive_name, ViveBase.feature_set_PVO), 'vive_PVO'),
            (ViveBase(vive_name, ViveBase.feature_set_PVOW), 'vive_PVOW'),
        ]

        for (base, name) in scenarios:
            self.supervisor.recognizer.initialize(base.feature_count, base.class_count, base.max_length)
            print('experiment 1 ' + name)
            self.run_user_dependent(name, base)
            self.run_user_independent(name, base)
            #self.run_cross_validate_user_independent(name)

        def experiment_2(self):
        # effect of gesture complexity
        old_classes = [
            'SwipeR',
            'SwipeDR',
            'SwipeD',
            'SwipeDL',
            'SwipeL',
            'SwipeUL',
            'SwipeU',
            'SwipeUR',
            'PokeR',
            'PokeU',
            'PokeL',
            'PokeD',
            'CircleHorizontalCCW',
            'CircleHorizontalCW',
            'CircleVerticalCCW',
            'CircleVerticalCW',
            'ShapeV',
            'ShapeX',
            'TwistL',
            'TwistR'
        ]
        new_classes = [
            'Square',
            'Teeth',
            'Triangle',
            'Infinity',
            'At',
            'Ampersand',
            'Brace',
            'Phi',
            'Gamma',
            'Heart'
        ]

        scenarios = [
            (ViveBase(vive_name, ViveBase.feature_set_PVOW, class_filter=old_classes), 'vive_PVOW'),
            (ViveBase(vive_name, ViveBase.feature_set_PVOW, class_filter=new_classes), 'vive_PVOW'),
            (ViveBase(vive_name, ViveBase.feature_set_PVOW, class_filter=None), 'vive_PVOW'),
        ]

        for (base, name) in scenarios:
            self.supervisor.recognizer.initialize(base.feature_count, base.class_count, base.max_length)
            print('experiment 2 ' + name)
            self.run_user_dependent(name, base)
            self.run_user_independent(name, base)
            #self.run_cross_validate_user_independent(name)


    def experiment_3(self):
        # determine effect of number of features
        scenarios = [
            (WiiBase(wii_name, WiiBase.feature_set_P), 'wii_P'),
            (WiiBase(wii_name, WiiBase.feature_set_PO), 'wii_PO'),
            (WiiBase(wii_name, WiiBase.feature_set_A), 'wii_A'),
            (WiiBase(wii_name, WiiBase.feature_set_AW), 'wii_AW'),
            (WiiBase(wii_name, WiiBase.feature_set_POAW), 'wii_POAW'),
            (ViveBase(vive_name, ViveBase.feature_set_POVW), 'vive_POVW'),
            (ViveBase(vive_name, ViveBase.feature_set_V), 'vive_V'),
            (ViveBase(vive_name, ViveBase.feature_set_VW), 'vive_VW'),
            (ViveBase(vive_name, ViveBase.feature_set_P), 'vive_P'),
            (ViveBase(vive_name, ViveBase.feature_set_PO), 'vive_PO'),
        ]

        for (base, name) in scenarios:
            self.supervisor.recognizer.initialize(base.feature_count, base.class_count, base.max_length)
            print('experiment 3 ' + name)
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
        all_sets = base.get_user_dependent_sets()
        i = 1
        for sets in all_sets:

            train_set, test_set, tester = sets
            print("UD {} / {}".format(i, len(all_sets)))
            i += 1
            self.supervisor.train_nn(train_set)
            self.supervisor.test_nn(test_set, analyzer)

        analyzer.save(name + '/user_dependent')

    def run_off_hand(self, name, base):
        analyzer = Analyzer(base.classes)

        train_set, test_set = base.get_other_hand_sets()
        self.supervisor.train_nn(train_set)
        self.supervisor.test_nn(test_set, analyzer)

        analyzer.save(name + '/off_hand')

if __name__ == '__main__':
    app = MyApp()
    app.experiment_1()
    app.experiment_2()
    app.experiment_3()
