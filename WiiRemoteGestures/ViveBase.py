import random
import sqlite3
import numpy as np

from SQLBase import SQLBase
from Gesture import Gesture


class ViveBase(SQLBase):
    def load(self, file):
        db = sqlite3.connect(file)
        c = db.cursor()

        command = 'SELECT class, user, trial, userHand, gestureHand, data FROM Gestures ORDER BY user '
            
        raw_list = c.execute(command).fetchall()

        for g in raw_list:
            name = g[0]
            self.testers.add(g[1])
            self.users[g[1]] = 'left' if g[3] == 0 else 'right'
            if name not in self.gesture_id:
                identifier = self.class_count
                self.class_count += 1
                self.gesture_name.append(name)
                self.gesture_id[name] = identifier

        row_size = 55

        for g in raw_list:
            data = np.frombuffer(g[5], dtype='float32')
            data = np.reshape(data, [-1, row_size])
            
            if g[4] == 0: # gesture hand is left
                data = data[:, 19:37]
            else:
                data = data[:, 37:55]

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