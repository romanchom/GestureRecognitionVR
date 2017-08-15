import random
import numpy as np
from Gesture import Gesture


class SQLBase:
    def __init__(self, has_none_class = False):
        self.gesture_name = []
        self.gesture_id = {}
        self.classes = self.gesture_name # alias
        if has_none_class:
            self.gesture_name.append('none')
            self.gesture_id['none'] = 0

        self.gesture_list = []
        self.testers = set()
        self.users = {}
        self.class_count = len(self.classes)
        self.max_length = 0
        self.feature_count = 0

        
    def load(self, file):
        raise NotImplementedError()

    def get_large_sets(self):
        '''Returns two disjoint sets:
            each containing half of gestures of each type of each tester'''
        train_set = []
        test_set = []
        for g in self.gesture_list:
            (train_set if g.trial % 2 == 0 else test_set).append(g)
        
        print("Train set: {}, Test set: {}".format(len(train_set), len(test_set)))
        return train_set, test_set

    def get_user_dependent_sets(self):
        '''Returns list of tuples of two disjoint sets:  
            each containing half of gestureas of each type of a single tester'''
        sets = []
        for tester in self.testers:
            train_set = []
            test_set = []
            for g in self.gesture_list:
                if(g.tester == tester):
                    (train_set if g.trial % 2 == 0 else test_set).append(g)
                    
            sets.append((train_set, test_set, tester))
        return sets

    def get_user_independent_sets(self, count):
        '''Returns two disjoints sets:
            one containing all gestures of random 5 righthanded testers,
            the other all the other gstures'''
        if len(self.users) < count:
            return self.get_cross_validation_sets()

        sets = []
        for i in range(count):
            all_users = list(self.users.items())
            random.shuffle(all_users)
            users = set()
            count = 5
            for user in all_users:
                if user[1] == 'right' or len(self.users) < 10:
                    users.add(user[0])
                    count -= 1
                
                if count == 0: break

            train_set = []
            test_set = []
            for g in self.gesture_list:
                if g.user_hand == 'right':
                    (train_set if g.tester in users else test_set).append(g)

            sets.append((train_set, test_set))
        return sets

    def get_other_hand_sets(self):
        """
        Returns a tuple of sets
        one contains gestures made with users' main hand
        other gestures made with users' off hand
        """
        train_set = []
        test_set = []
        for g in self.gesture_list:
            (train_set if g.user_hand == g.gesture_hand else test_set).append(g)
        
        return (train_set, test_set)

    def get_cross_validation_sets(self):
        """
        Returns list of tuples (training set, test set)
        where test set contains gestures of a single user
        and train set contains all other gestures
        """
        sets = []

        for tester in self.testers:
            print(tester)
            train_set = []
            test_set = []
            for gesture in self.gesture_list:
                (test_set if gesture.tester == tester else train_set).append(gesture)
            sets.append((train_set, test_set))

        return sets
                    
        

