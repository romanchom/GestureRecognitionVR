import random
import sqlite3
import numpy as np

from SQLBase import SQLBase
from Gesture import Gesture


class ViveBase(SQLBase):
    def __init__(self, file, feature_set_extractor = None, class_filter = None):
        super(ViveBase, self).__init__()
        db = sqlite3.connect(file)
        c = db.cursor()

        command = 'SELECT class, user, trial, userHand, gestureHand, data FROM Gestures ORDER BY user '
            
        raw_list = c.execute(command).fetchall()

        for g in raw_list:
            name = g[0]
            if class_filter != None:
                if name not in class_filter:
                    continue

            self.testers.add(g[1])
            self.users[g[1]] = 'left' if g[3] == 0 else 'right'
            if name not in self.gesture_id:
                identifier = self.class_count
                self.class_count += 1
                self.gesture_name.append(name)
                self.gesture_id[name] = identifier

        for g in raw_list:
            if g[0] not in self.gesture_id:
                continue

            data = np.frombuffer(g[5], dtype='float32')
            data = np.reshape(data, [-1, 55])
            
            if g[4] == 0: # gesture hand is left
                data = data[:, 19:37]
            else:
                data = data[:, 37:55]

            if feature_set_extractor:
                data = np.apply_along_axis(feature_set_extractor, 1, data)
            
            self.max_length = max(self.max_length, data.shape[0])

            gesture = Gesture(
                label_id = self.gesture_id[g[0]],
                tester = g[1],
                trial = g[2] - 1,
                user_hand = 'left' if g[3] == 0 else 'right',
                gesture_hand = 'left' if g[4] == 0 else 'right',
                data = data
            )
            self.gesture_list.append(gesture)

        self.feature_count = self.gesture_list[0].data.shape[1]

        for g in self.gesture_list:
            g.pad(self.max_length)

    def feature_set_P(row):
        return row[[3, 7, 11]]

    def feature_set_V(row):
        return row[12:15]

    def feature_set_O(row):
        return row[[0, 1, 2, 4, 5, 6, 8, 9, 10]]

    def feature_set_W(row):
        return row[15:18]

    def feature_set_PV(row):
        return row[[3, 7, 11, 12, 13, 14]]

    def feature_set_PO(row):
        return row[0:12]

    def feature_set_PVO(row):
        return row[0:15]

    def feature_set_VW(row):
        return row[12:18]

    feature_set_PVOW = None