import numpy as np

class Gesture:
    def __init__(self, label_id, tester, trial, user_hand, gesture_hand, data):
        self.label_id = label_id
        self.tester = tester
        self.trial = trial
        self.user_hand = user_hand
        self.gesture_hand = gesture_hand
        
        self.data = data
        self.length = data.shape[0]

    def pad(self, length):
        pad_amount = length - self.length
        self.data = np.pad(self.data, ((0, pad_amount), (0, 0)), 'edge')

    def to_numpy(gestures):
        count = len(gestures)
        shape = np.append(count, gestures[0].data.shape)

        data = np.ndarray(shape, 'float32')
        for i in range(count):
            data[i] = gestures[i].data
        
        labels = np.ndarray(count, 'int32')
        for i in range(count):
            labels[i] = gestures[i].label_id
        
        lengths = np.ndarray(count, 'int32')
        for i in range(count):
            lengths[i] = gestures[i].length
        
        return data, labels, lengths
        