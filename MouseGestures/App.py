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
from kivy.graphics import Color, Line, Ellipse


import tensorflow as tf
import numpy as np

maxLen = 60

class Gesture:
	def __init__(self):
		self.classId = 0
		self.points = []
	
	def from_points(self, input):
		for i in range(2, len(input), 2):
			self.points.append((input[i] - input[i-2], input[i+1]-input[i-1]))
			
	def to_dict(self):
		dict = {
			"classId" : self.classId,
			"points" : self.points
		}
		return dict
		
	def from_dict(dict):
		g = Gesture()
		g.classId = dict["classId"]
		g.points = dict["points"]
		return g
		
	def to_tensor(self):
		return self.points[:maxLen] + [self.points[-1]]*(maxLen - len(self.points))
		

class GestureBase:
	def __init__(self):
		self.gestures = []
		self.gestureIds = {'None': 0}
	
	def save(self, path):
		print("Gestures %d" % len(self.gestures))
		with open(path, 'w') as file:
			data = {
				"classes": self.gestureIds,
				"gestures": [g.to_dict() for g in self.gestures], 
			}
			json.dump(data, file, indent=2)
			
	def load(self, path):
		with open(path, 'r') as file:
			data = json.load(file)
			self.gestureIds = data["classes"]
			self.gestures = [Gesture.from_dict(g) for g in data["gestures"]]
	
	def get_classes_in_order(self):
		items = sorted(self.gestureIds.items(), key=lambda p: p[1])
		return [i[0] for i in items]
	
	def add_gesture(self, className, points):
		gesture = Gesture()
		if className not in self.gestureIds:
			self.gestureIds[className] = gesture.classId = len(self.gestureIds)
		else:
			gesture.classId = self.gestureIds[className]
		
		gesture.from_points(points)
		self.gestures.append(gesture)
		
	def to_tensor(self):
		return np.array([g.to_tensor() for g in self.gestures])
		
	def classes_to_tensor(self):
		ret = []
		for g in self.gestures:
			list = [0] * 10
			list[g.classId] = 1
			ret.append(list)
		return np.array(ret)
		
	def lengths_to_tensor(self):
		ret = [len(g.points) for g in self.gestures]
		return np.array(ret)
		#return [100 for g in self.gestures]
		
class GestureRecognizer:
	
	numberOfExamples = None #dynamic
	sampleVectorLen = 2 # x, y coords
	numMemCells = 24
	
	
	def __init__(self):
		self.inputData = tf.placeholder(tf.float32, [None, maxLen, self.sampleVectorLen])
		self.expectedClasses = tf.placeholder(tf.float32, [None, 10])
		self.inputLengths = tf.placeholder(tf.int32, [None])
				
		cell = tf.contrib.rnn.LSTMCell(self.numMemCells, state_is_tuple=True)
		cellOut, cellState = tf.nn.dynamic_rnn(
			cell, self.inputData, dtype=tf.float32, sequence_length=self.inputLengths)
		
		#last = cellState[1]
		#cellOut = tf.transpose(cellOut, [1, 0, 2])
		batchSize = tf.shape(cellOut)[0]
		index = tf.range(0, batchSize) * maxLen + (self.inputLengths - 1)
		flat = tf.reshape(cellOut, [-1, self.numMemCells])
		last = tf.gather(flat, index)
		print(last.get_shape())
		#last = tf.gather(cellOutTrans, int(cellOutTrans.get_shape()[0]) - 1)
		
		weight = tf.Variable(tf.truncated_normal([self.numMemCells, int(self.expectedClasses.get_shape()[1])], stddev = 0.1))
		bias = tf.Variable(tf.constant(0.1, shape=[self.expectedClasses.get_shape()[1]]))
		#prediction = tf.nn.softmax(tf.matmul(last, weight) + bias)
		prediction = tf.matmul(last, weight) + bias
		
		cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=prediction, labels=self.expectedClasses))
		#cross_entropy = -tf.reduce_sum(self.expectedClasses * tf.log(tf.clip_by_value(prediction,1e-10,1.0)))
		optimizer = tf.train.GradientDescentOptimizer(0.1)
		self.trainer = optimizer.minimize(cross_entropy)
		
		correct_prediction = tf.equal(tf.argmax(prediction,1), tf.argmax(self.expectedClasses,1))
		self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
		
		self.predictionMax = tf.argmax(prediction, 1)
		self.classifier = tf.nn.softmax(prediction)
		
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
		index, prob = self.sess.run([self.predictionMax, self.classifier], feed)
		
		index = index[0]
		print("Class Id %d" % index)
		prob = prob[0][index]
		print("Probability {:.1%}".format(prob))
		

class ToolBar(BoxLayout):	
	def __init__(self, **kwargs):
		self.controller = kwargs.pop("controller")
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
		
		self.loadButton = Button(text='Load Gestures')
		self.loadButton.bind(on_release=self.load_gestures)
		self.add_widget(self.loadButton)
		
		self.learnButton = Button(text='Learn')
		self.learnButton.bind(on_release=self.learn_gestures)
		self.add_widget(self.learnButton)
		
		self.toggleModeButton = Button(text='Adding')
		self.toggleModeButton.bind(on_release=self.toggle_mode)
		self.add_widget(self.toggleModeButton)
		
		
	def save_gestures(self, button):
		self.controller.save()
		
	def load_gestures(self, button):
		self.controller.load()
		
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
		self.p = (touch.x, touch.y)
		if self.collide_point(*self.p):
			self.canvas.clear()
			color = (random(), 1, 1)
			with self.canvas:
				Color(*color, mode='hsv')
				touch.ud['line'] = Line(points=[touch.x, touch.y])
				Ellipse(pos=[touch.x - 2, touch.y - 2], size=[4, 4])

	def on_touch_move(self, touch):
		if 'line' in touch.ud:
			p = [touch.x, touch.y]
			if ((np.linalg.norm(np.subtract(p, self.p))) > 20):
				with self.canvas:
					Ellipse(pos=[touch.x - 2, touch.y - 2], size=[4, 4])
				line = touch.ud['line']
				line.points += p
				self.p = p
		
	def on_touch_up(self, touch):
		if 'line' in touch.ud:
			line = touch.ud['line']
			
			self.previousLine = line
			self.controller.handle_gesture(line.points)
			print(len(line.points))


class GestureApp(App):

	def build(self):
		layout = BoxLayout(orientation='vertical')
		
		self.toolBar = ToolBar(size_hint=(1, None), height=40, controller=self)
		layout.add_widget(self.toolBar)
		
		mainArea = MyPaintWidget(size_hint=(1, 1))
		mainArea.controller = self
		layout.add_widget(mainArea)
		
		return layout

	def clear_canvas(self, obj):
		self.painter.canvas.clear()
		
		
	def __init__(self, **kwargs):
		super(GestureApp, self).__init__(**kwargs)
		self.handle_gesture = self.add_gesture_from_points
		self.gestureRecognizer = GestureRecognizer()
		self.base = GestureBase()
		self.className = 'None'
	
	def add_gesture_from_points(self, points):
		self.base.add_gesture(self.className, points)
		
	def save(self):
		self.base.save("Gestures.json")
		
	def load(self):
		self.base.load("Gestures.json")
		asd = self.base.get_classes_in_order()
		print(asd)
		self.toolBar.classesSpinner.values = self.base.get_classes_in_order()
		
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


if __name__ == '__main__':
	GestureApp().run()