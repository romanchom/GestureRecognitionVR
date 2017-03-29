import sqlite3
import numpy as np
from itertools import groupby
from Gesture import Gesture
from Example import Example
import random

class SQLBase:
    def __init__(self, file):
        self.gesture_name = ['none']
        self.gesture_id = {'none' : 0}
        self.gesture_list = []
        self.class_count = 1
        self.max_length = 0

        db = sqlite3.connect(file)
        c = db.cursor()
        command = 'SELECT name, tester, trial, righthand, data FROM GestureTable ORDER BY name'
        raw_list = c.execute(command).fetchall()

        for g in raw_list:
            name = g[0]
            if name not in self.gesture_id:
                identifier = self.class_count
                self.class_count += 1
                self.gesture_name.append(name)
                self.gesture_id[name] = identifier
                
        features = np.array([1, 2, 3, 4, 5, 6, 7])
        self.feature_count = len(features)
        for g in raw_list:
            data = np.reshape(np.frombuffer(g[4], dtype='float32'), [-1, 14])[:, features]
            self.max_length = max(self.max_length, data.shape[0])
            self.gesture_list.append(Gesture(self.gesture_id[g[0]], g[1], g[2] - 1, g[3], data))

        # 14 floats per time point
        # [0] timestamp
        # [1:4] xyz position in meters
        # [4:8] quaternion orientation
        # [8:11] acceleration in local (controller) space
        # [11:14] angular speed (yaw, pitch, roll)
        
    def get_large_sets(self):
        '''Returns two disjoint sets:
            each containing half of gestures of each type of each tester'''
        train_set = []
        test_set = []
        for g in self.gesture_list:
            #(train_set if g.trial < 5 else test_set).append((g.label_id, g.data))
            (train_set if g.trial < 5 else test_set).append(Example(g.data, g.label_id, self.max_length, 0.5))
        return train_set, test_set

    def get_user_dependent_sets(self, tester):
        '''Returns two disjoint sets:  
            each containing half of gestureas of each type of a single tester'''
        user = random.choice(self.gesture_list).tester
        train_set = []
        test_set = []
        for g in self.gesture_list:
            if(g.tester == user):
                (train_set if g.trial < 5 else test_set).append(Example(g.data, g.label_id, self.max_length, 0.5))
        return train_set, test_set

    def get_user_independent_sets(self):
        '''Returns two disjoints sets:
            one containing all gestures of random 5 righthanded testers,
            the other all the other gstures'''
        pass
