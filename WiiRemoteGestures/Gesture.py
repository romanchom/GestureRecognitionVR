

class Gesture:
    def __init__(self, label_id, tester, trial, hand, data):
        self.label_id = label_id
        self.tester = tester
        self.trial = trial
        self.hand = hand
        self.data = data

    def __repr__(self):
        return '<Gesture(ID={} T={} #{} H={}>'.format(self.label_id, self.tester, self.trial, self.hand)