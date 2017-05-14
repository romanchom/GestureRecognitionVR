import random
import sqlite3
from itertools import groupby

import numpy as np

from Example import Example
from Gesture import Gesture


class SQLBase:
    def __init__(self, file, mode):
        self.gesture_name = ['none']
        self.gesture_id = {'none' : 0}
        self.gesture_list = []
        self.users = {}
        self.class_count = 1
        self.max_length = 0

        db = sqlite3.connect(file)
        c = db.cursor()

        row_size = 55
        command = 'SELECT class, user, trial, userHand, gestureHand, data FROM Gestures ORDER BY user '
            
        raw_list = c.execute(command).fetchall()

        for g in raw_list:
            name = g[0]
            self.users[g[1]] = g[3]
            if name not in self.gesture_id:
                identifier = self.class_count
                self.class_count += 1
                self.gesture_name.append(name)
                self.gesture_id[name] = identifier

        for g in raw_list:
            data = np.frombuffer(g[5], dtype='float32')
            data = np.reshape(data, [-1, row_size])
            gestureHand = g[4]
            if gestureHand == 0: # lefthanded
                data = data[:, 19:37]
            else:
                data = data[:, 37:55]
            self.max_length = max(self.max_length, data.shape[0])
            self.gesture_list.append(Gesture(self.gesture_id[g[0]], g[1], g[2] - 1, 1 - g[3], data))

        self.feature_count = self.gesture_list[0].data.shape[1]
        
    def get_large_sets(self):
        '''Returns two disjoint sets:
            each containing half of gestures of each type of each tester'''
        train_set = []
        test_set = []
        for g in self.gesture_list:
            (train_set if g.trial % 2 == 0 else test_set).append(Example(g.data, g.label_id, self.max_length))
        
        print("Train set: {}, Test set: {}".format(len(train_set), len(test_set)))
        return train_set, test_set

    def get_user_dependent_sets(self, tester):
        '''Returns two disjoint sets:  
            each containing half of gestureas of each type of a single tester'''
        train_set = []
        test_set = []
        for g in self.gesture_list:
            if(g.tester == tester):
                (train_set if g.trial % 2 == 0 else test_set).append(Example(g.data, g.label_id, self.max_length))
                
        print("Train set: {}, Test set: {}".format(len(train_set), len(test_set)))
        return train_set, test_set

    def get_user_independent_sets(self):
        '''Returns two disjoints sets:
            one containing all gestures of random 5 righthanded testers,
            the other all the other gstures'''
        all_users = list(self.users.items())
        random.shuffle(all_users)
        users = set()
        count = 2
        for user in all_users:
            #if user[1] == 1:
            users.add(user[0])
            count -= 1
            
            if count == 0: break

        train_set = []
        test_set = []
        for g in self.gesture_list:
            if g.hand == 1:
                (train_set if g.tester in users else test_set).append(Example(g.data, g.label_id, self.max_length))

        print("Train set: {}, Test set: {}".format(len(train_set), len(test_set)))
        return train_set, test_set


