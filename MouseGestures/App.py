from random import random
from functools import partial
import csv
import json

from kivy.app import App
from kivy.uix.widget import Widget
from kivy.uix.button import Button
from kivy.uix.label import Label
from kivy.uix.textinput import TextInput
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.spinner import Spinner
from kivy.graphics import Color, Line


import tensorflow as tf
import numpy as np

class Gesture:
	def __init__(self):
		self.classId = 0
		self.points = []
	
	def from_points(self, input):
		for i in range(2, len(input), 2):
			self.points.append((input[i] - input[i-2], input[i+1]-input[i-1]))
			
	def to_dict(self):
		return {"class" : self.classId, "points" : self.points}
		
	def to_tensor(self):
		return self.points[:100] + [self.points[-1]]*(100 - len(self.points))
		

class GestureBase:
	gestures = []
	gestureIds = {'None': 0}
	
	def save(self, path):
		print("Gestures %d" % len(self.gestures))
		with open(path, 'w') as file:
			json.dump([g.to_dict() for g in self.gestures], file, indent=2)
		
	def load(self, path):
		pass
		
	def add_gesture(self, className, points):
		gesture = Gesture()
		if className not in self.gestureIds:
			self.gestureIds[className] = gesture.classId = len(self.gestureIds)
		else:
			gesture.classId = self.gestureIds[className]
		
		gesture.from_points(points)
		self.gestures.append(gesture)
		
	def to_tensor(self):
		return [g.to_tensor() for g in self.gestures]
		
	def classes_to_tensor(self):
		ret = []
		for g in self.gestures:
			list = [0] * 10
			list[g.classId] = 1
			ret.append(list)
		return ret
		
	def lengths_to_tensor(self):
		#return [len(g.points) for g in self.gestures]
		return [100 for g in self.gestures]
		
class GestureRecognizer:
	
	numberOfExamples = None #dynamic
	maxGestureLength = 100
	sampleVectorLen = 2 # x, y coords
	numMemCells = 24
	
	
	def __init__(self):
		self.inputData = tf.placeholder(tf.float32, [None, self.maxGestureLength, self.sampleVectorLen])
		self.expectedClasses = tf.placeholder(tf.float32, [None, 10])
		self.inputLengths = tf.placeholder(tf.int32, [None])
		print("########################")
		print(tf.shape(self.inputData)[0])
				
		cell = tf.nn.rnn_cell.LSTMCell(self.numMemCells, state_is_tuple=True)
		cellOut, cellState = tf.nn.dynamic_rnn(
			cell, self.inputData, dtype=tf.float32, sequence_length=self.inputLengths)
		
		#last = cellState[1]
		#cellOut = tf.transpose(cellOut, [1, 0, 2])
		batchSize = tf.shape(cellOut)[0]
		index = tf.range(0, batchSize) * self.maxGestureLength + (self.inputLengths - 1)
		flat = tf.reshape(cellOut, [-1, self.numMemCells])
		last = tf.gather(flat, index)
		print(last.get_shape())
		#last = tf.gather(cellOutTrans, int(cellOutTrans.get_shape()[0]) - 1)
		
		weight = tf.Variable(tf.truncated_normal([self.numMemCells, int(self.expectedClasses.get_shape()[1])], stddev = 0.1))
		bias = tf.Variable(tf.constant(0.1, shape=[self.expectedClasses.get_shape()[1]]))
		prediction = tf.nn.softmax(tf.matmul(last, weight) + bias)
		
		cross_entropy = -tf.reduce_sum(self.expectedClasses * tf.log(tf.clip_by_value(prediction,1e-10,1.0)))
		optimizer = tf.train.GradientDescentOptimizer(0.1)
		self.trainer = optimizer.minimize(cross_entropy)
		
		correct_prediction = tf.equal(tf.argmax(prediction,1), tf.argmax(self.expectedClasses,1))
		self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
		
		self.predictionMax = tf.argmax(prediction, 1)
		self.classifier = prediction
		
		self.sess = tf.Session()
		init_op = tf.global_variables_initializer()
		self.sess.run(init_op)
		
	def train(self, base):
		examples = base.to_tensor()
		labels = base.classes_to_tensor()
		lengths = base.lengths_to_tensor()
		
		feed = {self.inputData: examples,
				self.expectedClasses: labels,
				self.inputLengths: lengths}
		
		for i in range(1000):
			self.sess.run(self.trainer, feed)
			if i % 10 == 0:
				print("Epoch %d" % i)
				print(self.sess.run(self.accuracy, feed))
				
		print("Trained")
				
	def classify(self, points):
		gesture = Gesture()
		gesture.from_points(points)
		
		feed = {self.inputData: [gesture.to_tensor()],
			self.inputLengths: [len(gesture.points)]}
		res, probs = self.sess.run([self.classifier, self.predictionMax], feed)
		
		print(res)
		print(probs)

class Controller:
	base = GestureBase()
	className = 'None'
	handle_gesture = None
	gestureRecognizer = None
	
	def __init__(self):
		self.handle_gesture = self.add_gesture_from_points
		self.gestureRecognizer = GestureRecognizer()
	
	def add_gesture_from_points(self, points):
		self.base.add_gesture(self.className, points)
		
	def save(self):
		self.base.save("Gestures.json")
		
	def learn(self):
		self.gestureRecognizer.train(self.base)
	
	def classify(self, points):
		self.gestureRecognizer.classify(points)
		
	def toggle_mode(self):
		if(self.handle_gesture == self.add_gesture_from_points):
			self.handle_gesture = self.classify
			return "Classifying"
		else:
			self.handle_gesture = self.add_gesture_from_points
			return "Adding"
		

class ToolBar(BoxLayout):
	controller = None
	classInput = None
	classesSpinner = None
	saveButton = None
	learnButton = None
	toggleModeButton = None
	
	def __init__(self, **kwargs):
		super(ToolBar, self).__init__(**kwargs)
		
		self.add_widget(Label(text='Class:'))
		
		self.classesSpinner = Spinner(values=['None'], text='None')
		self.classesSpinner.bind(text=self.select_class)
		self.add_widget(self.classesSpinner)
		
		self.add_widget(Label(text='Add new'))
		self.classInput = TextInput(multiline=False)
		self.classInput.bind(on_text_validate=self.add_class)
		self.add_widget(self.classInput)
		
		self.saveButton = Button(text='Save Gestures')
		self.saveButton.bind(on_release=self.save_gestures)
		self.add_widget(self.saveButton)
		
		self.learnButton = Button(text='Learn')
		self.learnButton.bind(on_release=self.learn_gestures)
		self.add_widget(self.learnButton)
		
		self.toggleModeButton = Button(text='Adding')
		self.toggleModeButton.bind(on_release=self.toggle_mode)
		self.add_widget(self.toggleModeButton)
		
		
	def save_gestures(self, button):
		self.controller.save()
		
	def learn_gestures(self, button):
		self.controller.learn()
		
	def classify_gestures(self, button):
		self.controller.classify()
	
	def toggle_mode(self, button):
		button.text = self.controller.toggle_mode()
		
	def add_class(self,  text):
		self.classesSpinner.values.append(text.text)
		text.text = ''
		
	def select_class(self, spinner, text):
		self.controller.className = text
		


class MyPaintWidget(Widget):
	previousLine = None
	controller = None
	
	def __init__(self, **kwargs):
		super(MyPaintWidget, self).__init__(**kwargs)
	
	def on_touch_down(self, touch):
		p = (touch.x, touch.y)
		if self.collide_point(*p):
			color = (random(), 1, 1)
			with self.canvas:
				Color(*color, mode='hsv')
				touch.ud['line'] = Line(points=[touch.x, touch.y])

	def on_touch_move(self, touch):
		if 'line' in touch.ud:
			line = touch.ud['line']
			line.points += [touch.x, touch.y]
		
	def on_touch_up(self, touch):
		if 'line' in touch.ud:
			line = touch.ud['line']
			if(self.previousLine != None):
				self.canvas.remove(self.previousLine)
			
			self.previousLine = line
			self.controller.handle_gesture(line.points)



class GestureApp(App):
	controller = Controller()

	def build(self):
		layout = BoxLayout(orientation='vertical')
		
		toolbar = ToolBar(size_hint=(1, None), height=40)
		toolbar.controller = self.controller
		layout.add_widget(toolbar)
		
		mainArea = MyPaintWidget(size_hint=(1, 1))
		mainArea.controller = self.controller
		layout.add_widget(mainArea)
		
		return layout

	def clear_canvas(self, obj):
		self.painter.canvas.clear()


if __name__ == '__main__':
	GestureApp().run()