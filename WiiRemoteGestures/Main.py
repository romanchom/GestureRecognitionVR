from SQLBase import SQLBase
from Recognizer import Recognizer



base = SQLBase('MotionGesture.db')
recognizer = Recognizer(base.feature_count, base.class_count, base.max_length)
