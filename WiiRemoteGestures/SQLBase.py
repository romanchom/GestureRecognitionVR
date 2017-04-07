import random
import sqlite3
from itertools import groupby

import numpy as np

from Example import Example
from Gesture import Gesture


class SQLBase:
    def __init__(self, file):
        self.gesture_name = ['none']
        self.gesture_id = {'none' : 0}
        self.gesture_list = []
        self.users = {}
        self.class_count = 1
        self.max_length = 0
        self.feature_count = 18

        db = sqlite3.connect(file)
        c = db.cursor()
        command = 'SELECT name, tester, trial, righthand, data FROM GestureTable ORDER BY name'
        raw_list = c.execute(command).fetchall()

        for g in raw_list:
            name = g[0]
            self.users[g[1]] =  g[3]
            if name not in self.gesture_id:
                identifier = self.class_count
                self.class_count += 1
                self.gesture_name.append(name)
                self.gesture_id[name] = identifier
                
        for g in raw_list:
            data = np.apply_along_axis(SQLBase.transform_data, 1, np.reshape(np.frombuffer(g[4], dtype='float32'), [-1, 14]))
            self.max_length = max(self.max_length, data.shape[0])
            self.gesture_list.append(Gesture(self.gesture_id[g[0]], g[1], g[2] - 1, g[3], data))

        # 14 floats per time point
        # [0] timestamp
        # [1:4] xyz position in meters
        # [4:8] quaternion orientation
        # [8:11] acceleration in local (controller) space
        # [11:14] angular speed (yaw, pitch, roll)
        
    def transform_data(data):
        ret = np.zeros(18, 'float32')
        # copy position
        ret[0:3] = data[1:4]

        # transform quaternion to rotation matrix
        # order of result actually doesn't matter
        qw = data[4]
        qx = data[5]
        qy = data[6]
        qz = data[7]

        sqw = qw * qw
        sqx = qx * qx
        sqy = qy * qy
        sqz = qz * qz
              
        ret[3] = sqx - sqy - sqz + sqw
        ret[7] = -sqx + sqy - sqz + sqw
        ret[11] = -sqx - sqy + sqz + sqw
        tmp1 = qx * qy
        tmp2 = qz * qw
        ret[6] = 2 * (tmp1 + tmp2)
        ret[4] = 2 * (tmp1 - tmp2)
        tmp1 = qw * qz
        tmp2 = qy * qw
        ret[9] = 2 * (tmp1 - tmp2)
        ret[5] = 2 * (tmp1 + tmp2)
        tmp1 = qy * qz
        tmp2 = qx * qw
        ret[10] = 2 * (tmp1 + tmp2)
        ret[8] = 2 * (tmp1 - tmp2)

        ret[12:18] = data[8:14]

        return ret


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
        train_set = []
        test_set = []
        for g in self.gesture_list:
            if(g.tester == tester):
                (train_set if g.trial < 5 else test_set).append(Example(g.data, g.label_id, self.max_length, 0.5))
        return train_set, test_set

    def get_user_independent_sets(self):
        '''Returns two disjoints sets:
            one containing all gestures of random 5 righthanded testers,
            the other all the other gstures'''
        all_users = list(self.users.items())
        random.shuffle(all_users)
        users = set()
        count = 5
        for user in all_users:
            if user[1] == 1:
                users.add(user[0])
                count -= 1
            
            if count == 0: break

        train_set = []
        test_set = []
        for g in self.gesture_list:
            (train_set if g.tester in users else test_set).append(Example(g.data, g.label_id, self.max_length, 0.5))
        return train_set, test_set


