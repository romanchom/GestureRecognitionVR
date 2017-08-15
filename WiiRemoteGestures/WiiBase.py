import random
import sqlite3
import numpy as np

from SQLBase import SQLBase
from Gesture import Gesture


class WiiBase(SQLBase):
    def __init__(self, file, feature_set_extractor = None):
        super(WiiBase, self).__init__()

        db = sqlite3.connect(file)
        c = db.cursor()
        command = 'SELECT name, tester, trial, righthand, data FROM GestureTable WHERE trial <= 10 ORDER BY name '
        raw_list = c.execute(command).fetchall()

        for g in raw_list:
            name = g[0]
            self.testers.add(g[1])
            self.users[g[1]] =  'left' if g[3] == 0 else 'right'
            if name not in self.gesture_id:
                identifier = self.class_count
                self.class_count += 1
                self.gesture_name.append(name)
                self.gesture_id[name] = identifier

        for g in raw_list:
            data = np.frombuffer(g[4], dtype='float32')
            data = np.reshape(data, [-1, 14])

            if feature_set_extractor:
                #data = np.apply_along_axis(feature_set_extractor, 1, data)
                data = feature_set_extractor(data)

            self.max_length = max(self.max_length, data.shape[0])
            gesture = Gesture(
                label_id = self.gesture_id[g[0]],
                tester = g[1],
                trial = g[2] - 1,
                user_hand = 'left' if g[3] == 0 else 'right',
                gesture_hand = 'left' if g[3] == 0 else 'right',
                data = data
            )
            self.gesture_list.append(gesture)

        self.feature_count = self.gesture_list[0].data.shape[1]

        for g in self.gesture_list:
            g.pad(self.max_length)
        #print(self.gesture_list[0].data[:, 1])
        # 14 floats per time point
        # [0] timestamp
        # [1:4] xyz position in meters
        # [4:8] quaternion orientation
        # [8:11] acceleration in local (controller) space
        # [11:14] angular speed (yaw, pitch, roll)

    def feature_set_P(data):
        return np.apply_along_axis(lambda row: row[1:4], 1, data)

    def feature_set_V(data):
        shape = list(data.shape)
        shape[1] = 3
        vel = np.zeros(shape, dtype='float32')
        count = shape[0]
        for i in range(1, count):
            vel[i] = (data[i, 1:4] - data[i - 1, 1:4]) * 60

        return vel

    def feature_set_PV(data):
        pos = WiiBase.feature_set_P(data)
        vel = WiiBase.feature_set_V(data)
        return np.concatenate((pos, vel), 1)

    def feature_set_PO(data):
        return np.apply_along_axis(lambda row: row[1:8], 1, data)

    def feature_set_O(data):
        return np.apply_along_axis(lambda row: row[4:8], 1, data)

    def feature_set_W(data):
        return np.apply_along_axis(lambda row: row[4:8], 1, data)

    def feature_set_A(data):
        return np.apply_along_axis(lambda row: row[8:11], 1, data)

    def feature_set_AW(data):
        return np.apply_along_axis(lambda row: row[8:14], 1, data)
    
    def feature_set_AWO(data):
        return np.apply_along_axis(lambda row: row[4:14], 1, data)

    def feature_set_PVO(data):
        po = WiiBase.feature_set_PO(data)
        vel = WiiBase.feature_set_V(data)
        return np.concatenate((po, vel), 1)

    def feature_set_PVOW(data):
        po = WiiBase.feature_set_PO(data)
        w = np.apply_along_axis(lambda row: row[11:14], 1, data)
        vel = WiiBase.feature_set_V(data)
        return np.concatenate((po, w, vel), 1)

    def feature_set_PVOWA(data):
        poaw = WiiBase.feature_set_POAW(data)
        vel = WiiBase.feature_set_V(data)
        return np.concatenate((poaw, vel), 1)

    def feature_set_POAW(data):
        return np.apply_along_axis(lambda row: row[1:14], 1, data)