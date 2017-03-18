from SQLBase import SQLBase
from Example import Example
import numpy as np
from Recognizer import Recognizer



base = SQLBase('MotionGesture.db')
recognizer = Recognizer(base.feature_count, base.class_count, base.max_length)
train_set, test_set = base.get_large_sets()

Example.to_numpy(train_set)




